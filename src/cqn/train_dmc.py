import argparse
import os
import time
import warnings
from pathlib import Path

warnings.filterwarnings('ignore', category=DeprecationWarning, module='torch.utils.data')
os.environ['MUJOCO_GL'] = 'osmesa'

import numpy as np
import torch

import src.cqn.dmc as dmc
from src.common.checkpoint_manager import CheckpointManager
from src.common.logger import Logger
from src.common.metrics_tracker import MetricsTracker
from src.common.training_utils import init_training
from src.cqn.agent import CQNAgent
from src.cqn.replay_buffer_dmc import ReplayBufferStorage, make_replay_loader
from src.cqn.video import TrainVideoRecorder, VideoRecorder

torch.backends.cudnn.benchmark = True


class TrainingConfig:
    """Configuration for training CQN agent."""

    def __init__(self, args):
        for key, value in vars(args).items():
            setattr(self, key.replace("-", "_"), value)

    @staticmethod
    def from_args():
        """Create configuration from command line arguments."""
        parser = argparse.ArgumentParser(description="Train CQN Agent on DMC")

        TrainingConfig._add_task_arguments(parser)
        TrainingConfig._add_training_arguments(parser)
        TrainingConfig._add_evaluation_arguments(parser)
        TrainingConfig._add_replay_buffer_arguments(parser)
        TrainingConfig._add_agent_arguments(parser)
        TrainingConfig._add_cqn_arguments(parser)
        TrainingConfig._add_logging_arguments(parser)

        args = parser.parse_args()
        return TrainingConfig(args)

    @staticmethod
    def _add_task_arguments(parser):
        parser.add_argument("--task-name", type=str, default="walker_walk")
        parser.add_argument("--frame-stack", type=int, default=3)
        parser.add_argument("--action-repeat", type=int, default=2)
        parser.add_argument("--discount", type=float, default=0.99)

    @staticmethod
    def _add_training_arguments(parser):
        parser.add_argument("--num-train-frames", type=int, default=1100000)
        parser.add_argument("--num-seed-frames", type=int, default=4000)
        parser.add_argument("--seed", type=int, default=1)

    @staticmethod
    def _add_evaluation_arguments(parser):
        parser.add_argument("--eval-every-frames", type=int, default=10000)
        parser.add_argument("--num-eval-episodes", type=int, default=10)

    @staticmethod
    def _add_replay_buffer_arguments(parser):
        parser.add_argument("--replay-buffer-size", type=int, default=1000000)
        parser.add_argument("--replay-buffer-num-workers", type=int, default=4)
        parser.add_argument("--nstep", type=int, default=1)
        parser.add_argument("--low-dim-obs-shape", type=int, default=1)
        parser.add_argument("--batch-size", type=int, default=512)

    @staticmethod
    def _add_agent_arguments(parser):
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--weight-decay", type=float, default=0.1)
        parser.add_argument("--feature-dim", type=int, default=64)
        parser.add_argument("--hidden-dim", type=int, default=512)
        parser.add_argument("--critic-target-tau", type=float, default=0.02)
        parser.add_argument("--update-every-steps", type=int, default=2)
        parser.add_argument("--num-expl-steps", type=int, default=2000)

    @staticmethod
    def _add_cqn_arguments(parser):
        parser.add_argument("--levels", type=int, default=3)
        parser.add_argument("--bins", type=int, default=5)
        parser.add_argument("--atoms", type=int, default=51)
        parser.add_argument("--v-min", type=float, default=0)
        parser.add_argument("--v-max", type=float, default=200)
        parser.add_argument("--critic-lambda", type=float, default=0.1)
        parser.add_argument("--stddev-schedule", type=float, default=0.1)
        parser.add_argument("--bc-lambda", type=float, default=1.0)
        parser.add_argument("--bc-margin", type=float, default=0.01)

    @staticmethod
    def _add_logging_arguments(parser):
        parser.add_argument("--log-interval", type=int, default=5)
        parser.add_argument("--checkpoint-interval", type=int, default=50000)
        parser.add_argument("--save-video", action="store_true")
        parser.add_argument("--save-train-video", action="store_true")
        parser.add_argument("--save-snapshot", action="store_true")
        parser.add_argument("--use-tb", action="store_true")
        parser.add_argument("--use-wandb", action="store_true")
        parser.add_argument("--load-checkpoint", type=str, default=None)
        parser.add_argument("--resume", action="store_true")


class Timer:
    """Simple timer for tracking elapsed time."""

    def __init__(self):
        self._start_time = time.time()
        self._last_time = time.time()

    def reset(self):
        """Reset timer and return elapsed and total time."""
        elapsed_time = time.time() - self._last_time
        self._last_time = time.time()
        total_time = time.time() - self._start_time
        return elapsed_time, total_time

    def total_time(self):
        """Get total time since timer creation."""
        return time.time() - self._start_time


class AgentFactory:
    """Factory for creating CQN agents."""

    @staticmethod
    def create(obs_spec, action_spec, config, device):
        """Create a CQN agent with the given specifications."""
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


class EnvironmentManager:
    """Manages training and evaluation environments."""

    def __init__(self, config):
        self.config = config
        self.train_env = None
        self.eval_env = None
        self._initialize_environments()

    def _initialize_environments(self):
        """Initialize both training and evaluation environments."""
        self.train_env = self._create_environment()
        self.eval_env = self._create_environment()

    def _create_environment(self):
        """Create a single DMC environment."""
        return dmc.make(
            self.config.task_name,
            self.config.frame_stack,
            self.config.action_repeat,
            self.config.seed,
        )

    def get_observation_spec(self):
        """Get observation specification from training environment."""
        return self.train_env.observation_spec()

    def get_action_spec(self):
        """Get action specification from training environment."""
        return self.train_env.action_spec()


class ReplayBufferManager:
    """Manages replay buffer storage and loading."""

    def __init__(self, config, working_dir, data_specs):
        self.config = config
        self.working_dir = working_dir
        self.storage = ReplayBufferStorage(data_specs, working_dir / "buffer")
        self.loader = self._create_loader()
        self._replay_iter = None

    def _create_loader(self):
        """Create replay buffer data loader."""
        return make_replay_loader(
            self.working_dir / "buffer",
            self.config.replay_buffer_size,
            self.config.batch_size,
            self.config.replay_buffer_num_workers,
            self.config.save_snapshot,
            self.config.nstep,
            self.config.discount,
            self.config.low_dim_obs_shape,
        )

    def get_iterator(self):
        """Get or create replay buffer iterator."""
        if self._replay_iter is None:
            self._replay_iter = iter(self.loader)
        return self._replay_iter

    def add(self, time_step):
        """Add time step to replay buffer."""
        self.storage.add(time_step)

    def __len__(self):
        return len(self.storage)


class VideoManager:
    """Manages video recording for training and evaluation."""

    def __init__(self, working_dir, save_video, save_train_video):
        self.eval_recorder = VideoRecorder(
            working_dir if save_video else None
        )
        self.train_recorder = TrainVideoRecorder(
            working_dir if save_train_video else None
        )

    def init_eval_recording(self, env, enabled):
        """Initialize evaluation video recording."""
        self.eval_recorder.init(env, enabled=enabled)

    def init_train_recording(self, observation):
        """Initialize training video recording."""
        self.train_recorder.init(observation)

    def record_eval_frame(self, env):
        """Record a single evaluation frame."""
        self.eval_recorder.record(env)

    def record_train_frame(self, observation):
        """Record a single training frame."""
        self.train_recorder.record(observation)

    def save_eval_video(self, filename):
        """Save evaluation video."""
        self.eval_recorder.save(filename)

    def save_train_video(self, filename):
        """Save training video."""
        self.train_recorder.save(filename)


class EpisodeMetrics:
    """Tracks metrics for individual episodes."""

    def __init__(self):
        self.step = 0
        self.reward = 0.0
        self.recent_rewards = []
        self.recent_lengths = []

    def reset(self):
        """Reset episode metrics."""
        self.step = 0
        self.reward = 0.0

    def update_step(self, reward):
        """Update metrics for a single step."""
        self.reward += reward
        self.step += 1

    def finalize_episode(self, action_repeat):
        """Finalize episode and store metrics."""
        episode_length = self.step * action_repeat
        self.recent_rewards.append(self.reward)
        self.recent_lengths.append(episode_length)

    def should_log(self, episode, log_interval):
        """Check if metrics should be logged."""
        return episode % log_interval == 0 and len(self.recent_rewards) > 0

    def get_averages(self):
        """Get average reward and length."""
        avg_reward = np.mean(self.recent_rewards) if self.recent_rewards else 0
        avg_length = np.mean(self.recent_lengths) if self.recent_lengths else 0
        return avg_reward, avg_length

    def clear_recent(self):
        """Clear recent metrics."""
        self.recent_rewards = []
        self.recent_lengths = []


class CQNTrainer(Logger):
    """Main trainer for CQN agent on DMC tasks."""

    def __init__(self, config, working_dir="./src/cqn/output"):
        super().__init__(working_dir + "/logs")
        self.config = config
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)

        self.device = self._get_device()
        self.timer = Timer()
        self._global_step = 0
        self._global_episode = 0

        self._initialize_components()
        self._load_checkpoint_if_needed()

    def _get_device(self):
        """Determine and return the compute device."""
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _initialize_components(self):
        """Initialize all training components."""
        self._initialize_managers()
        self._initialize_environments()
        self._initialize_agent()
        init_training(self.config.seed, self.device, self.logger)

    def _initialize_managers(self):
        """Initialize checkpoint, metrics, and video managers."""
        self.checkpoint_manager = CheckpointManager(
            logger=self.logger,
            checkpoint_dir=str(self.working_dir / "checkpoints")
        )
        self.metrics_tracker = MetricsTracker(
            logger=self.logger,
            save_dir=str(self.working_dir / "metrics")
        )

    def _initialize_environments(self):
        """Initialize environment and replay buffer."""
        self.env_manager = EnvironmentManager(self.config)

        data_specs = self._create_data_specs()
        self.replay_manager = ReplayBufferManager(
            self.config,
            self.working_dir,
            data_specs
        )

        self.video_manager = VideoManager(
            self.working_dir,
            self.config.save_video,
            self.config.save_train_video
        )

    def _create_data_specs(self):
        """Create data specifications for replay buffer."""
        from dm_env import specs
        return (
            self.env_manager.get_observation_spec(),
            self.env_manager.get_action_spec(),
            specs.Array((1,), np.float32, "reward"),
            specs.Array((1,), np.float32, "discount"),
        )

    def _initialize_agent(self):
        """Initialize the CQN agent."""
        self.agent = AgentFactory.create(
            self.env_manager.get_observation_spec(),
            self.env_manager.get_action_spec(),
            self.config,
            self.device
        )

    def _load_checkpoint_if_needed(self):
        """Load checkpoint if resume or load_checkpoint is specified."""
        if self.config.resume or self.config.load_checkpoint:
            self._load_checkpoint()

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self._global_step * self.config.action_repeat

    def train(self):
        """Main training loop."""
        episode_metrics = EpisodeMetrics()
        time_step = self._initialize_training_episode(episode_metrics)
        training_metrics = None

        while self._should_continue_training():
            if time_step.last():
                time_step = self._handle_episode_end(episode_metrics, training_metrics)
                episode_metrics.reset()

            if self._should_evaluate():
                self._run_evaluation()

            if self._should_save_checkpoint():
                self._save_checkpoint()

            action = self._select_action(time_step)

            if self._should_update_agent():
                training_metrics = self._update_agent()
                self._log_training_metrics(training_metrics)

            time_step = self._execute_action(action, episode_metrics)

        self._finalize_training()

    def _should_continue_training(self):
        """Check if training should continue."""
        return self.global_frame < self.config.num_train_frames

    def _should_evaluate(self):
        """Check if evaluation should run."""
        return self.global_frame % self.config.eval_every_frames == 0

    def _should_save_checkpoint(self):
        """Check if checkpoint should be saved."""
        return self.global_frame % self.config.checkpoint_interval == 0

    def _should_update_agent(self):
        """Check if agent should be updated."""
        return self.global_frame >= self.config.num_seed_frames

    def _initialize_training_episode(self, episode_metrics):
        """Initialize the first training episode."""
        time_step = self.env_manager.train_env.reset()
        self.replay_manager.add(time_step)
        self.video_manager.init_train_recording(time_step.observation)
        episode_metrics.reset()
        return time_step

    def _handle_episode_end(self, episode_metrics, training_metrics):
        """Handle end of episode tasks."""
        self._global_episode += 1
        self.video_manager.save_train_video(f"{self.global_frame}.mp4")

        if training_metrics is not None:
            self._process_episode_metrics(episode_metrics)

        if self.config.save_snapshot:
            self._save_snapshot()

        time_step = self.env_manager.train_env.reset()
        self.replay_manager.add(time_step)
        self.video_manager.init_train_recording(time_step.observation)

        return time_step

    def _process_episode_metrics(self, episode_metrics):
        """Process and log episode metrics."""
        elapsed_time, total_time = self.timer.reset()
        episode_metrics.finalize_episode(self.config.action_repeat)

        if episode_metrics.should_log(self.global_episode, self.config.log_interval):
            self._log_episode_metrics(episode_metrics, elapsed_time, total_time)
            episode_metrics.clear_recent()

    def _log_episode_metrics(self, episode_metrics, elapsed_time, total_time):
        """Log episode metrics to logger."""
        avg_reward, avg_length = episode_metrics.get_averages()
        total_frames = sum(episode_metrics.recent_lengths)
        fps = total_frames / elapsed_time if elapsed_time > 0 else 0

        self.logger.info("=== Training Progress ===")
        self.logger.info(f"Frame: {self.global_frame}")
        self.logger.info(f"Episode: {self.global_episode}")
        self.logger.info(f"Step: {self.global_step}")
        self.logger.info(f"Avg reward (last {len(episode_metrics.recent_rewards)} eps): {avg_reward:.2f}")
        self.logger.info(f"Avg length: {avg_length:.2f}")
        self.logger.info(f"FPS: {fps:.2f}")
        self.logger.info(f"Buffer size: {len(self.replay_manager)}")
        self.logger.info(f"Total time: {total_time:.2f}s")

        self.metrics_tracker.log_episode(
            episode=self.global_episode,
            reward=avg_reward,
            steps=int(avg_length),
            episode_time=total_time
        )

    def _select_action(self, time_step):
        """Select action for current time step."""
        with torch.no_grad():
            self.agent.eval()
            low_dim_obs = self._create_dummy_low_dim_obs()
            action = self.agent.act(
                time_step.observation,
                low_dim_obs,
                self.global_step,
                eval_mode=False
            )
            self.agent.train()
        return action

    def _create_dummy_low_dim_obs(self):
        """Create dummy low-dimensional observation."""
        return np.zeros(self.config.low_dim_obs_shape, dtype=np.float32)

    def _update_agent(self):
        """Update agent with batch from replay buffer."""
        return self.agent.update(
            self.replay_manager.get_iterator(),
            self.global_step
        )

    def _log_training_metrics(self, metrics):
        """Log training metrics if available."""
        if metrics and self.global_step % (self.config.log_interval * 10) == 0:
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            self.logger.info(f"Training metrics - {metrics_str}")

    def _execute_action(self, action, episode_metrics):
        """Execute action in environment and record results."""
        time_step = self.env_manager.train_env.step(action)
        episode_metrics.update_step(time_step.reward)
        self.replay_manager.add(time_step)
        self.video_manager.record_train_frame(time_step.observation)
        self._global_step += 1
        return time_step

    def _run_evaluation(self):
        """Run evaluation episodes."""
        self.logger.info(f"Starting evaluation at step {self.global_step}")

        total_reward = 0.0
        total_steps = 0

        for episode in range(self.config.num_eval_episodes):
            reward, steps = self._run_single_evaluation_episode(episode)
            total_reward += reward
            total_steps += steps

        self._log_evaluation_results(total_reward, total_steps)

    def _run_single_evaluation_episode(self, episode_index):
        """Run a single evaluation episode."""
        time_step = self.env_manager.eval_env.reset()
        self.video_manager.init_eval_recording(
            self.env_manager.eval_env,
            enabled=(episode_index == 0)
        )

        episode_reward = 0.0
        episode_steps = 0

        while not time_step.last():
            action = self._select_evaluation_action(time_step)
            time_step = self.env_manager.eval_env.step(action)
            self.video_manager.record_eval_frame(self.env_manager.eval_env)
            episode_reward += time_step.reward
            episode_steps += 1

        if episode_index == 0:
            self.video_manager.save_eval_video(f"{self.global_frame}.mp4")

        return episode_reward, episode_steps

    def _select_evaluation_action(self, time_step):
        """Select action during evaluation."""
        with torch.no_grad():
            self.agent.eval()
            low_dim_obs = self._create_dummy_low_dim_obs()
            action = self.agent.act(
                time_step.observation,
                low_dim_obs,
                self.global_step,
                eval_mode=True
            )
            self.agent.train()
        return action

    def _log_evaluation_results(self, total_reward, total_steps):
        """Log evaluation results."""
        num_episodes = self.config.num_eval_episodes
        avg_reward = total_reward / num_episodes
        avg_length = (total_steps * self.config.action_repeat) / num_episodes

        self.logger.info(f"=== Evaluation at frame {self.global_frame} ===")
        self.logger.info(f"Average episode reward: {avg_reward:.2f}")
        self.logger.info(f"Average episode length: {avg_length:.2f}")
        self.logger.info(f"Episode: {self.global_episode}")
        self.logger.info(f"Step: {self.global_step}")

        self.metrics_tracker.log_episode(
            episode=self.global_episode,
            reward=avg_reward,
            steps=int(avg_length),
            episode_time=self.timer.total_time()
        )

    def _save_checkpoint(self):
        """Save training checkpoint."""
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            agent=self,
            episode=self.global_episode,
            task_name=self.config.task_name,
            seed=self.config.seed
        )
        self.logger.info(f"Checkpoint saved at frame {self.global_frame}: {checkpoint_path}")

    def _load_checkpoint(self):
        """Load checkpoint from file."""
        checkpoint_path = self._get_checkpoint_path()

        if checkpoint_path:
            self._restore_from_checkpoint(checkpoint_path)
        else:
            self.logger.info("No checkpoint found, starting fresh training")

    def _get_checkpoint_path(self):
        """Get checkpoint path for loading."""
        if self.config.resume and not self.config.load_checkpoint:
            return self.checkpoint_manager.find_latest_checkpoint()
        return self.config.load_checkpoint

    def _restore_from_checkpoint(self, checkpoint_path):
        """Restore training state from checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path)

            self._restore_agent_state(checkpoint)
            self._restore_training_state(checkpoint)

            self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
            self.logger.info(
                f"Resuming from episode {self.global_episode}, "
                f"step {self.global_step}, frame {self.global_frame}"
            )
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            self.logger.info("Starting from scratch...")

    def _restore_agent_state(self, checkpoint):
        """Restore agent state from checkpoint."""
        self.agent.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.agent.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.agent.encoder_opt.load_state_dict(checkpoint['encoder_opt_state_dict'])
        self.agent.critic_opt.load_state_dict(checkpoint['critic_opt_state_dict'])

    def _restore_training_state(self, checkpoint):
        """Restore training state from checkpoint."""
        self._global_step = checkpoint['global_step']
        self._global_episode = checkpoint['global_episode']
        self.timer = checkpoint.get('timer', Timer())

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

    def _save_snapshot(self):
        """Save snapshot of current training state."""
        snapshot = self.working_dir / "snapshot.pt"
        keys_to_save = ["agent", "timer", "_global_step", "_global_episode"]
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open("wb") as f:
            torch.save(payload, f)

    def _save_metrics(self):
        """Save training metrics."""
        self.metrics_tracker.save_metrics(
            agent="CQN",
            task_name=self.config.task_name,
            seed=self.config.seed
        )

    def _finalize_training(self):
        """Finalize training and save final state."""
        self.logger.info("Training complete! Saving final checkpoint and metrics...")
        self._save_checkpoint()
        self._save_metrics()


def main():
    """Main entry point for training."""
    config = TrainingConfig.from_args()
    trainer = CQNTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
