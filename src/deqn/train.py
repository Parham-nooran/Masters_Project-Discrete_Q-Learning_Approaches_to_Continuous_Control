import argparse
import gc
import time

import torch

from src.common.checkpoint_manager import CheckpointManager
from src.common.logger import Logger
from src.common.metrics_accumulator import MetricsAccumulator
from src.common.metrics_tracker import MetricsTracker
from src.common.training_utils import (
    process_observation,
    get_env_specs,
    get_env,
    init_training,
)
from src.deqn.agent import DecQNAgent
from src.deqn.config import create_config_from_args
from src.plotting.plotting_utils import PlottingUtils


def parse_args():
    parser = argparse.ArgumentParser(description="Train Decoupled Q-Networks Agent")
    parser.add_argument(
        "--load-checkpoints",
        type=str,
        default=None,
        help="Path to checkpoints file to resume from",
    )
    parser.add_argument(
        "--load-metrics",
        type=str,
        default=None,
        help="Path to metrics file to resume from",
    )
    parser.add_argument(
        "--task", type=str, default="reacher", help="Environment task"
    )
    parser.add_argument(
        "--num-episodes", type=int, default=1000, help="Number of episodes to train"
    )
    parser.add_argument(
        "--num-steps", type=int, default=1000000, help="Number of steps to train"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--mock-episode-length",
        type=int,
        default=100,
        help="Mock episode length for training",
    )
    parser.add_argument(
        "--algorithm", type=str, default="decqnvis", help="Algorithm to use"
    )
    parser.add_argument(
        "--num-bins", type=int, default=2, help="Number of bins for discretization"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.1, help="Epsilon for exploration"
    )
    parser.add_argument("--discount", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument(
        "--target-update-period",
        type=int,
        default=100,
        help="Target network update period",
    )
    parser.add_argument(
        "--min-replay-size", type=int, default=1000, help="Minimum replay buffer size"
    )
    parser.add_argument(
        "--max-replay-size", type=int, default=500000, help="Maximum replay buffer size"
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Save checkpoints every N episodes",
    )
    parser.add_argument(
        "--log-interval", type=int, default=5, help="Log progress every N episodes"
    )
    parser.add_argument(
        "--detailed-log-interval",
        type=int,
        default=50,
        help="Detailed log every N episodes",
    )
    return parser.parse_args()


def _convert_action_to_numpy(action):
    """Convert action tensor to numpy array."""
    if isinstance(action, torch.Tensor):
        return action.cpu().numpy()
    return action


def _update_agent_parameters(agent):
    """Update agent parameters like epsilon."""
    agent.update_epsilon(decay_rate=0.995, min_epsilon=0.01)


class DecQNTrainer(Logger):
    """Trainer for Decoupled Q-Networks Agent."""

    def __init__(self, config, working_dir="./src/deqn/output"):
        super().__init__(working_dir + "/logs")
        self.working_dir = working_dir
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.checkpoint_manager = CheckpointManager(self.logger, checkpoint_dir=self.working_dir + "/checkpoints")

    def train(self):
        """Execute main training loop."""
        self._setup_training()

        env = get_env(self.config.task, self.logger)
        obs_shape, action_spec_dict = get_env_specs(env, self.config.use_pixels)
        agent = DecQNAgent(self.config, obs_shape, action_spec_dict)

        start_episode = self.checkpoint_manager.load_checkpoint_if_available(
            self.config.load_checkpoints, agent
        )
        metrics_tracker = self._initialize_metrics_tracker(start_episode, save_dir=self.working_dir + "/metrics")

        self._log_setup_info(agent)

        self._run_training_loop(env, agent, metrics_tracker, start_episode)
        self._finalize_training(agent, metrics_tracker)

        return agent

    def _setup_training(self):
        """Initialize training environment."""
        init_training(self.config.seed, self.device, self.logger)

    def _log_setup_info(self, agent):
        """Log training setup information."""
        self.logger.info("Decoupled Q-Networks Agent Setup:")
        self.logger.info(f"  Task: {self.config.task}")
        self.logger.info(f"  Decouple: {agent.config.decouple}")
        self.logger.info(f"  Action dimensions: {agent.action_discretizer.action_dim}")

    def _run_training_loop(self, env, agent, metrics_tracker, start_episode):
        """Execute the main training loop."""
        metrics_accumulator = MetricsAccumulator()
        start_time = time.time()

        for episode in range(start_episode, self.config.num_episodes):
            episode_metrics = self._run_episode(env, agent, metrics_accumulator)
            self._log_episode_metrics(episode, episode_metrics, start_time)
            _update_agent_parameters(agent)
            self._perform_periodic_maintenance(episode)
            self._save_checkpoint_if_needed(agent, metrics_tracker, episode)

            metrics_tracker.log_episode(episode=episode, **episode_metrics)

    def _run_episode(self, env, agent, metrics_accumulator):
        """Run a single training episode."""
        episode_start_time = time.time()
        episode_reward = 0.0
        steps = 0

        time_step = env.reset()
        obs = process_observation(
            time_step.observation, self.config.use_pixels, self.device
        )
        agent.observe_first(obs)

        while not time_step.last() and steps < 1000:
            action = agent.select_action(obs)
            action_np = _convert_action_to_numpy(action)

            time_step = env.step(action_np)
            next_obs = process_observation(
                time_step.observation, self.config.use_pixels, self.device
            )
            reward = time_step.reward if time_step.reward is not None else 0.0
            done = time_step.last()

            agent.observe(action, reward, next_obs, done)

            self._update_networks_if_ready(agent, metrics_accumulator)

            obs = next_obs
            episode_reward += reward
            steps += 1

        episode_time = time.time() - episode_start_time
        averages = metrics_accumulator.get_averages()

        return {
            "reward": episode_reward,
            "steps": steps,
            "loss": averages["loss"],
            "mean_abs_td_error": averages["mean_abs_td_error"],
            "mean_squared_td_error": averages["mean_squared_td_error"],
            "q_mean": averages["q_mean"],
            "epsilon": agent.epsilon,
            "mse_loss": averages["mse_loss"],
            "episode_time": episode_time,
        }

    def _update_networks_if_ready(self, agent, metrics_accumulator):
        """Update networks if replay buffer has enough samples."""
        if len(agent.replay_buffer) <= self.config.min_replay_size:
            return

        metrics = agent.update()
        metrics_accumulator.update(metrics)

    def _log_episode_metrics(self, episode, metrics, start_time):
        """Log episode metrics at specified intervals."""
        if episode % self.config.log_interval == 0:
            self._log_basic_metrics(episode, metrics)

        if episode % self.config.detailed_log_interval == 0 and episode > 0:
            self._log_detailed_metrics(episode, start_time)

    def _log_basic_metrics(self, episode, metrics):
        """Log basic episode metrics."""
        self.logger.info(
            f"Episode {episode:4d} | "
            f"Num Steps {metrics['steps']:4d} | "
            f"Episodic Reward: {metrics['reward']:7.2f} | "
            f"Loss: {metrics['loss']:8.6f} | "
            f"MSE Loss: {metrics['mse_loss']:8.6f} | "
            f"Mean abs TD Error: {metrics['mean_abs_td_error']:8.6f} | "
            f"Mean squared TD Error: {metrics['mean_squared_td_error']:8.6f} | "
            f"Q-mean: {metrics['q_mean']:6.3f} | "
            f"Time: {metrics['episode_time']:.2f}s | "
            f"Buffer: {len(self.agent.replay_buffer) if hasattr(self, 'agent') else 0:6d}"
        )

    def _log_detailed_metrics(self, episode, start_time):
        """Log detailed training progress."""
        elapsed_time = time.time() - start_time
        episodes_completed = episode + 1
        avg_episode_time = elapsed_time / episodes_completed
        remaining_episodes = self.config.num_episodes - episode - 1
        eta = avg_episode_time * remaining_episodes

        self.logger.info(f"Episode {episode} Summary:")
        self.logger.info(
            f"Elapsed: {elapsed_time / 60:.1f}min | ETA: {eta / 60:.1f}min"
        )

    def _perform_periodic_maintenance(self, episode):
        """Perform periodic memory cleanup."""
        if episode % 10 == 0 and self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

        if self.device == "cuda":
            torch.cuda.synchronize()

    def _save_checkpoint_if_needed(self, agent, metrics_tracker, episode):
        """Save checkpoint at specified intervals."""
        if episode % self.config.checkpoint_interval != 0:
            return

        metrics_tracker.save_metrics()
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            agent, episode, self.config.task
        )
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

    def _finalize_training(self, agent, metrics_tracker):
        """Finalize training by saving and plotting."""
        metrics_tracker.save_metrics()

        final_checkpoint = self.checkpoint_manager.save_checkpoint(
            agent, self.config.num_episodes, self.config.task + "_final"
        )
        self.logger.info(f"Final checkpoint saved: {final_checkpoint}")

        self._generate_plots(metrics_tracker)

    def _generate_plots(self, metrics_tracker):
        """Generate training plots."""
        self.logger.info("Generating plots...")
        plotter = PlottingUtils(self.logger, metrics_tracker, self.working_dir + "/plots")
        plotter.plot_training_curves(save=True)
        plotter.plot_reward_distribution(save=True)
        plotter.print_summary_stats()

    def _initialize_metrics_tracker(self, start_episode, save_dir):
        """Initialize or load metrics tracker."""
        metrics_tracker = MetricsTracker(self.logger, save_dir)

        if start_episode > 0:
            metrics_tracker.load_metrics(self.config.load_metrics)

        return metrics_tracker


if __name__ == "__main__":
    args = parse_args()
    config = create_config_from_args(parse_args())
    trainer = DecQNTrainer(config)
    trained_agent = trainer.train()
