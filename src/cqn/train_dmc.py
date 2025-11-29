import argparse
from pathlib import Path
from types import SimpleNamespace
import os

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='torch.utils.data')

# os.environ['MUJOCO_GL'] = 'osmesa'


import numpy as np
import torch
from dm_env import specs

import src.cqn.dmc as dmc
import src.cqn.utils as utils
from src.common.checkpoint_manager import CheckpointManager
from src.common.logger import Logger
from src.common.metrics_tracker import MetricsTracker
from src.common.training_utils import init_training
from src.cqn.agent import CQNAgent
from src.cqn.replay_buffer_dmc import ReplayBufferStorage, make_replay_loader
from src.cqn.video import TrainVideoRecorder, VideoRecorder

torch.backends.cudnn.benchmark = True


def args2name_space(args):
    """Create config object from parsed arguments."""
    config = SimpleNamespace()
    for key, value in vars(args).items():
        setattr(config, key.replace("-", "_"), value)
    return config


def parse_args():
    parser = argparse.ArgumentParser(description="Train CQN Agent on DMC")

    # Task settings
    parser.add_argument("--task-name", type=str, default="walker_walk", help="DMC task name")
    parser.add_argument("--frame-stack", type=int, default=3, help="Number of frames to stack")
    parser.add_argument("--action-repeat", type=int, default=2, help="Action repeat")
    parser.add_argument("--discount", type=float, default=0.99, help="Discount factor")

    # Training settings
    parser.add_argument("--num-train-frames", type=int, default=1100000, help="Number of training frames")
    parser.add_argument("--num-seed-frames", type=int, default=4000, help="Number of seed frames")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")

    # Evaluation
    parser.add_argument("--eval-every-frames", type=int, default=10000, help="Evaluate every N frames")
    parser.add_argument("--num-eval-episodes", type=int, default=10, help="Number of evaluation episodes")

    # Replay buffer
    parser.add_argument("--replay-buffer-size", type=int, default=1000000, help="Replay buffer size")
    parser.add_argument("--replay-buffer-num-workers", type=int, default=4, help="Number of replay buffer workers")
    parser.add_argument("--nstep", type=int, default=1, help="N-step returns")
    parser.add_argument("--low-dim-obs-shape", type=int, default=1, help="low dim obs shape")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")

    # Agent parameters
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--feature-dim", type=int, default=64, help="Feature dimension")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--critic-target-tau", type=float, default=0.02, help="Critic target tau")
    parser.add_argument("--update-every-steps", type=int, default=2, help="Update every N steps")
    parser.add_argument("--num-expl-steps", type=int, default=2000, help="Number of exploration steps")

    # CQN specific
    parser.add_argument("--levels", type=int, default=3, help="Number of levels")
    parser.add_argument("--bins", type=int, default=5, help="Number of bins")
    parser.add_argument("--atoms", type=int, default=51, help="Number of atoms")
    parser.add_argument("--v-min", type=float, default=0, help="Minimum value")
    parser.add_argument("--v-max", type=float, default=200, help="Maximum value")
    parser.add_argument("--critic-lambda", type=float, default=0.1, help="Critic Lambda")
    parser.add_argument("--stddev-schedule", type=float, default=0.1, help="Stddev schedule")
    parser.add_argument("--bc-lambda", type=float, default=1.0, help="bc lambda")
    parser.add_argument("--bc-margin", type=float, default=0.01, help="bc margin")

    # Logging and checkpointing
    parser.add_argument("--log-interval", type=int, default=5, help="Log every N steps")
    parser.add_argument("--checkpoint-interval", type=int, default=50000,
                        help="Save checkpoint every N frames")
    parser.add_argument("--save-video", action="store_true", help="Save videos")
    parser.add_argument("--save-train-video", action="store_true", help="Save training videos")
    parser.add_argument("--use-tb", action="store_true", help="Use tensorboard")
    parser.add_argument("--use-wandb", action="store_true", help="Use wandb")

    # Checkpoint and metrics loading
    parser.add_argument("--load-checkpoint", type=str, default=None, help="Path to checkpoint file")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")

    return parser.parse_args()


def make_agent(obs_spec, action_spec, config, device):
    """Create CQN agent with the given specs and config."""
    return CQNAgent(
        low_dim_obs_shape=config.low_dim_obs_shape,
        bc_lambda=config.bc_lambda,
        bc_margin=config.bc_margin,
        critic_lambda=config.critic_lambda,
        weight_decay=config.weight_decay,
        rgb_obs_shape=obs_spec.shape,
        action_shape=action_spec.shape,
        device=device,
        lr=config.lr,
        critic_target_tau=config.critic_target_tau,
        update_every_steps=config.update_every_steps,
        use_logger=config.use_tb or config.use_wandb,
        num_expl_steps=config.num_expl_steps,
        feature_dim=config.feature_dim,
        hidden_dim=config.hidden_dim,
        levels=config.levels,
        bins=config.bins,
        atoms=config.atoms,
        v_min=config.v_min,
        v_max=config.v_max,
        stddev_schedule=config.stddev_schedule
    )


class CQNTrainer(Logger):
    def __init__(self, config, working_dir="./src/cqn/output"):
        super().__init__(working_dir + "/logs")
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = config
        self.agent_name = "cqn"
        self.checkpoint_manager = CheckpointManager(
            logger=self.logger,
            checkpoint_dir=str(self.working_dir / "checkpoints")
        )
        self.metrics_tracker = MetricsTracker(
            logger=self.logger,
            save_dir=str(self.working_dir / "metrics")
        )
        self.setup()
        self.agent = make_agent(
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            self.config,
            self.device
        )
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

        if self.config.resume or self.config.load_checkpoint:
            self.load_checkpoint()

    def setup(self):
        """Setup environments and replay buffer."""
        self.train_env = dmc.make(
            self.config.task_name,
            self.config.frame_stack,
            self.config.action_repeat,
            self.config.seed,
        )
        self.eval_env = dmc.make(
            self.config.task_name,
            self.config.frame_stack,
            self.config.action_repeat,
            self.config.seed,
        )
        data_specs = (
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            specs.Array((1,), np.float32, "reward"),
            specs.Array((1,), np.float32, "discount"),
        )

        self.replay_storage = ReplayBufferStorage(data_specs, self.working_dir / "buffer")

        self.replay_loader = make_replay_loader(
            self.working_dir / "buffer",
            self.config.replay_buffer_size,
            self.config.batch_size,
            self.config.replay_buffer_num_workers,
            self.config.nstep,
            self.config.discount,
            self.config.low_dim_obs_shape,
        )
        self._replay_iter = None

        self.video_recorder = VideoRecorder(
            self.working_dir if self.config.save_video else None
        )
        self.train_video_recorder = TrainVideoRecorder(
            self.working_dir if self.config.save_train_video else None
        )
        init_training(self.config.seed, self.device, self.logger)

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.config.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.config.num_eval_episodes)

        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    # Create dummy low_dim_obs with correct shape
                    low_dim_obs = np.zeros(self.config.low_dim_obs_shape, dtype=np.float32)
                    action = self.agent.act(
                        time_step.observation,
                        low_dim_obs,
                        self.global_step,
                        eval_mode=True
                    )
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save(f"{self.global_frame}.mp4")

        avg_reward = total_reward / episode
        avg_length = step * self.config.action_repeat / episode

        self.logger.info(f"=== Evaluation at frame {self.global_frame} ===")
        self.logger.info(f"Average episode reward: {avg_reward:.2f}")
        self.logger.info(f"Average episode length: {avg_length:.2f}")
        self.logger.info(f"Episode: {self.global_episode}")
        self.logger.info(f"Step: {self.global_step}")

        # Track evaluation metrics
        self.metrics_tracker.log_episode(
            episode=self.global_episode,
            reward=avg_reward,
            steps=int(avg_length),
            episode_time=self.timer.total_time()
        )

    def train(self):
        train_until_step = utils.Until(
            self.config.num_train_frames, self.config.action_repeat
        )
        seed_until_step = utils.Until(self.config.num_seed_frames, self.config.action_repeat)
        eval_every_step = utils.Every(
            self.config.eval_every_frames, self.config.action_repeat
        )

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        self.replay_storage.add(time_step)
        self.train_video_recorder.init(time_step.observation)
        metrics = None

        recent_rewards = []
        recent_lengths = []

        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                self.train_video_recorder.save(f"{self.global_frame}.mp4")

                if metrics is not None:
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.config.action_repeat

                    # Store for averaging
                    recent_rewards.append(episode_reward)
                    recent_lengths.append(episode_frame)
                    # Log every log_interval episodes
                    if self.global_episode % self.config.log_interval == 0:
                        self.log_episode(recent_rewards, recent_lengths, elapsed_time, total_time)
                        # Reset tracking
                        recent_rewards = []
                        recent_lengths = []

                time_step = self.train_env.reset()
                self.replay_storage.add(time_step)
                self.train_video_recorder.init(time_step.observation)

                episode_step = 0
                episode_reward = 0

            # Evaluation
            if eval_every_step(self.global_step):
                self.logger.info(f"Starting evaluation at step {self.global_step}")
                self.eval()

            if self._global_step % self.config.checkpoint_interval == 0:
                self.save_checkpoint()

            with torch.no_grad(), utils.eval_mode(self.agent):
                low_dim_obs = np.zeros(self.config.low_dim_obs_shape, dtype=np.float32)
                action = self.agent.act(
                    time_step.observation,
                    low_dim_obs,
                    self.global_step,
                    eval_mode=False
                )

            if not seed_until_step(self.global_step) and len(self.replay_storage) > 0:
                metrics = self.agent.update(self.replay_iter, self.global_step)
                if metrics and self.global_step % (self.config.log_interval * 10) == 0:
                    metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                    self.logger.info(f"Training metrics - {metrics_str}")

            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step)
            self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1

        self.logger.info("Training complete! Saving final checkpoint and metrics...")
        self.save_checkpoint()
        self.save_metrics()

    def save_checkpoint(self):
        """Save checkpoint with agent state and training progress."""
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            agent=self,
            episode=self.global_episode,
            task_name=self.config.task_name,
            seed=self.config.seed
        )
        self.logger.info(f"Checkpoint saved at frame {self.global_frame}: {checkpoint_path}")

    def load_checkpoint(self):
        """Load checkpoint from file."""
        checkpoint_path = self.config.load_checkpoint

        if self.config.resume and not checkpoint_path:
            checkpoint_path = self.checkpoint_manager.find_latest_checkpoint()

        if checkpoint_path:
            try:
                checkpoint = torch.load(checkpoint_path)

                # Load agent state
                self.agent.encoder.load_state_dict(checkpoint['encoder_state_dict'])
                self.agent.critic.load_state_dict(checkpoint['critic_state_dict'])
                self.agent.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
                self.agent.encoder_opt.load_state_dict(checkpoint['encoder_opt_state_dict'])
                self.agent.critic_opt.load_state_dict(checkpoint['critic_opt_state_dict'])

                # Load training state
                self._global_step = checkpoint['global_step']
                self._global_episode = checkpoint['global_episode']
                self.timer = checkpoint.get('timer', utils.Timer())

                self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
                self.logger.info(
                    f"Resuming from episode {self.global_episode}, step {self.global_step}, frame {self.global_frame}")

            except Exception as e:
                self.logger.error(f"Failed to load checkpoint: {e}")
                self.logger.info("Starting from scratch...")
        else:
            self.logger.info("No checkpoint found, starting fresh training")

    def get_checkpoint_state(self):
        """Get current state for checkpointing."""
        return {
            'encoder_state_dict': self.agent.encoder.state_dict(),
            'critic_state_dict': self.agent.critic.state_dict(),
            'critic_target_state_dict': self.agent.critic_target.state_dict(),
            'encoder_opt_state_dict': self.agent.encoder_opt.state_dict(),
            'critic_opt_state_dict': self.agent.critic_opt.state_dict(),
            'global_step': self._global_step,
            'global_episode': self._global_episode,
            'timer': self.timer,
            'config': self.config,
        }

    def save_metrics(self):
        """Save training metrics."""
        self.metrics_tracker.save_metrics(
            agent="CQN",
            task_name=self.config.task_name,
            seed=self.config.seed
        )

    def log_episode(self, recent_rewards, recent_lengths, elapsed_time, total_time):
        avg_reward = np.mean(recent_rewards)
        avg_length = np.mean(recent_lengths)
        fps = sum(recent_lengths) / elapsed_time if elapsed_time > 0 else 0
        self.metrics_tracker.save_metrics(agent=self.agent_name, task_name=self.config.task_name,
                                          seed=self.config.seed)

        self.logger.info(f"=== Training Progress ===")
        self.logger.info(f"Frame: {self.global_frame}")
        self.logger.info(f"Episode: {self.global_episode}")
        self.logger.info(f"Step: {self.global_step}")
        self.logger.info(f"Avg reward (last {len(recent_rewards)} eps): {avg_reward:.2f}")
        self.logger.info(f"Avg length: {avg_length:.2f}")
        self.logger.info(f"FPS: {fps:.2f}")
        self.logger.info(f"Buffer size: {len(self.replay_storage)}")
        self.logger.info(f"Total time: {total_time:.2f}s")
        self.logger.info(f"Action bin range: [{self.agent.critic.initial_low[0].item():.3f},"
                         f" {self.agent.critic.initial_high[0].item():.3f}]")
        self.metrics_tracker.log_episode(
            episode=self.global_episode,
            reward=avg_reward,
            steps=int(avg_length),
            episode_time=self.timer.total_time(),
            action_ranges=[{
                'low': self.agent.critic.initial_low.cpu().numpy().tolist(),
                'high': self.agent.critic.initial_high.cpu().numpy().tolist()
            }]
        )


if __name__ == "__main__":
    args = parse_args()
    config = args2name_space(args)
    trainer = CQNTrainer(config)
    trainer.train()