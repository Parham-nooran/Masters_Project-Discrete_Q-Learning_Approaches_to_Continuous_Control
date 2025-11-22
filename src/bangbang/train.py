import argparse
import time
from collections import deque

from src.bangbang.agent import BangBangAgent
from src.common.logger import Logger
from src.common.metrics_tracker import MetricsTracker
from src.common.training_utils import *
from src.common.checkpoint_manager import CheckpointManager


class BangBangTrainer(Logger):

    def __init__(self, args, working_dir="./src/bangbang/output"):
        super().__init__(working_dir + "/logs")
        self.working_dir = working_dir + "/" + args.algorithm
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.agent_name = f"bangbang_{self.args.algorithm}_"
        self.checkpoint_manager = CheckpointManager(
            self.logger,
            checkpoint_dir=self.working_dir + "/checkpoints"
        )

    def train(self):
        init_training(self.args.seed, self.device, self.logger)
        env = get_env(self.args.task, self.logger)
        obs_shape, action_spec_dict = get_env_specs(env, self.args.use_pixels)
        agent = BangBangAgent(self.args, obs_shape, action_spec_dict)
        start_episode = self.checkpoint_manager.load_checkpoint_if_available(
            self.args.resume_checkpoint,
            agent
        )
        metrics_tracker = MetricsTracker(self.logger, save_dir=self.working_dir + "/metrics")

        self._log_training_start(agent)

        start_time = time.time()

        for episode in range(start_episode, self.args.num_episodes):
            episode_metrics = self._run_episode(env, agent)

            metrics_tracker.log_episode(
                episode=episode,
                reward=episode_metrics["reward"],
                steps=episode_metrics["steps"],
                loss=episode_metrics["avg_loss"],
                mean_abs_td_error=0.0,
                mean_squared_td_error=0.0,
                q_mean=0.0,
                epsilon=0.0,
            )

            self._log_episode_progress(episode, episode_metrics, agent)
            self._log_detailed_progress(episode, metrics_tracker, start_time)
            self._save_checkpoint_if_needed(episode, agent, metrics_tracker)

        self._finalize_training(agent, metrics_tracker, start_time)

        return agent

    def _log_training_start(self, agent):
        self.logger.info(f"Starting Bang-Bang training on {self.args.task}")
        self.logger.info(f"Action dimension: {agent.action_dim}")
        self.logger.info(f"Algorithm: {self.args.algorithm}")
        self.logger.info(f"Device: {self.device}")

    def _run_episode(self, env, agent):
        episode_start_time = time.time()
        episode_reward = 0
        recent_losses = deque(maxlen=20)

        time_step = env.reset()
        obs = process_observation(
            time_step.observation, self.args.use_pixels, self.device
        )
        agent.observe_first(obs)

        steps = 0
        max_steps = getattr(self.args, "max_episode_steps", 1000)

        while not time_step.last() and steps < max_steps:
            action = agent.select_action(obs)

            time_step = self._execute_action(env, action)
            next_obs = process_observation(
                time_step.observation, self.args.use_pixels, self.device
            )
            reward = time_step.reward if time_step.reward is not None else 0.0
            done = time_step.last()

            agent.observe(action, reward, next_obs, done)

            if self._should_update(agent):
                metrics = agent.update()
                if metrics and "policy_loss" in metrics:
                    recent_losses.append(metrics["policy_loss"])

            obs = next_obs
            episode_reward += reward
            steps += 1

        return {
            "reward": episode_reward,
            "steps": steps,
            "avg_loss": self._compute_average_loss(recent_losses),
            "time": time.time() - episode_start_time,
        }

    def _execute_action(self, env, action):
        action_np = action.cpu().numpy()
        return env.step(action_np)

    def _should_update(self, agent):
        return len(agent.replay_buffer) > self.args.min_replay_size

    def _compute_average_loss(self, recent_losses):
        return np.mean(recent_losses) if recent_losses else 0.0

    def _log_episode_progress(self, episode, episode_metrics, agent):
        if episode % self.args.log_interval == 0:
            self.logger.info(
                f"Episode {episode:4d} | "
                f"Reward: {episode_metrics['reward']:7.2f} | "
                f"Loss: {episode_metrics['avg_loss']:8.6f} | "
                f"Time: {episode_metrics['time']:.2f}s | "
                f"Buffer: {len(agent.replay_buffer):6d}"
            )

    def _log_detailed_progress(self, episode, metrics_tracker, start_time):
        if episode % self.args.detailed_log_interval == 0 and episode > 0:
            elapsed_time = time.time() - start_time
            avg_episode_time = elapsed_time / episode
            eta = avg_episode_time * (self.args.num_episodes - episode)

            recent_rewards = metrics_tracker.episode_rewards[
                -self.args.detailed_log_interval :
            ]

            self.logger.info(f"Episode {episode} Summary:")
            self.logger.info(f"Recent avg reward: {np.mean(recent_rewards):.2f}")
            self.logger.info(f"ETA: {eta / 60:.1f} min")

    def _save_checkpoint_if_needed(self, episode, agent, metrics_tracker):
        if episode % self.args.checkpoint_interval == 0:
            self.checkpoint_manager.save_checkpoint(agent, episode, self.args.task, self.args.seed)
            metrics_tracker.save_metrics(self.agent_name, self.args.task, self.args.seed)

    def _finalize_training(self, agent, metrics_tracker, start_time):
        self.checkpoint_manager.save_checkpoint(agent, self.args.num_episodes, self.args.task, self.args.seed)
        metrics_tracker.save_metrics(self.agent_name, self.args.task, self.args.seed)
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time / 60:.1f} minutes!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Bang-Bang Control Agent")
    parser.add_argument(
        "--task", type=str, default="walker_walk", help="Environment task"
    )
    parser.add_argument(
        "--num-episodes", type=int, default=1000, help="Number of episodes"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--log-interval", type=int, default=10, help="Log interval")
    parser.add_argument(
        "--detailed-log-interval", type=int, default=50, help="Detailed log interval"
    )
    parser.add_argument(
        "--checkpoint-interval", type=int, default=100, help="Checkpoint interval"
    )

    # Network architecture
    parser.add_argument(
        "--use-pixels", action="store_true", help="Use pixel observations"
    )
    parser.add_argument(
        "--layer-size-network",
        type=int,
        nargs="+",
        default=[512, 512],
        help="Hidden layer sizes for policy/value networks",
    )
    parser.add_argument(
        "--layer-size-bottleneck", type=int, default=100, help="Encoder output size"
    )
    parser.add_argument(
        "--num-pixels", type=int, default=84, help="Pixel observation size"
    )

    # Training hyperparameters
    parser.add_argument(
        "--min-replay-size",
        type=int,
        default=1000,
        help="Minimum replay buffer size before training",
    )
    parser.add_argument(
        "--max-replay-size", type=int, default=500000, help="Maximum replay buffer size"
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--learning-rate", type=float, default=3e-4, help="Learning rate"
    )
    parser.add_argument("--discount", type=float, default=0.99, help="Discount factor")
    parser.add_argument(
        "--priority-exponent", type=float, default=0.6, help="Prioritized replay alpha"
    )
    parser.add_argument(
        "--importance-sampling-exponent",
        type=float,
        default=0.4,
        help="Prioritized replay beta",
    )
    parser.add_argument("--adder-n-step", type=int, default=1, help="N-step returns")
    parser.add_argument(
        "--clip-gradients", action="store_true", default=True, help="Clip gradients"
    )
    parser.add_argument(
        "--clip-gradients-norm", type=float, default=40.0, help="Gradient clipping norm"
    )
    parser.add_argument(
        "--max-episode-steps", type=int, default=1000, help="Maximum steps per episode"
    )

    # Algorithm selection
    parser.add_argument(
        "--algorithm",
        type=str,
        default="ppo",
        choices=["ppo", "sac", "mpo"],
        help="RL algorithm to use",
    )

    # Algorithm-specific parameters
    parser.add_argument(
        "--ppo-clip-ratio", type=float, default=0.2, help="PPO clipping ratio"
    )
    parser.add_argument(
        "--ppo-value-coef", type=float, default=0.5, help="PPO value loss coefficient"
    )
    parser.add_argument(
        "--sac-alpha", type=float, default=0.2, help="SAC entropy temperature"
    )
    parser.add_argument(
        "--sac-tau", type=float, default=0.005, help="SAC target network update rate"
    )
    parser.add_argument(
        "--mpo-epsilon", type=float, default=0.1, help="MPO KL constraint"
    )
    parser.add_argument(
        "--mpo-epsilon-penalty", type=float, default=0.001, help="MPO epsilon penalty"
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    args = parser.parse_args()

    trainer = BangBangTrainer(args)
    agent = trainer.train()
