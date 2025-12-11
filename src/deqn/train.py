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
    """Parse command line arguments for training configuration."""
    parser = argparse.ArgumentParser(description="Train Decoupled Q-Networks Agent")

    _add_checkpoint_arguments(parser)
    _add_environment_arguments(parser)
    _add_training_arguments(parser)
    _add_hyperparameter_arguments(parser)
    _add_logging_arguments(parser)

    return parser.parse_args()


def _add_checkpoint_arguments(parser):
    """Add checkpoint-related arguments."""
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


def _add_environment_arguments(parser):
    """Add environment-related arguments."""
    parser.add_argument(
        "--task", type=str, default="reacher_easy", help="Environment task"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--mock-episode-length",
        type=int,
        default=100,
        help="Mock episode length for training",
    )


def _add_training_arguments(parser):
    """Add training configuration arguments."""
    parser.add_argument(
        "--num-episodes", type=int, default=1000, help="Number of episodes to train"
    )
    parser.add_argument(
        "--num-steps", type=int, default=1000000, help="Number of steps to train"
    )
    parser.add_argument(
        "--algorithm", type=str, default="decqnvis", help="Algorithm to use"
    )


def _add_hyperparameter_arguments(parser):
    """Add hyperparameter arguments."""
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
        "--max-replay-size", type=int, default=1000000, help="Maximum replay buffer size"
    )


def _add_logging_arguments(parser):
    """Add logging and checkpointing arguments."""
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1000,
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
        self.device = self._initialize_device()
        self.agent_name = "deqn"

        self.checkpoint_manager = self._create_checkpoint_manager()
        self.env = self._create_environment()
        self.agent = self._create_agent()

    def _initialize_device(self):
        """Initialize computation device."""
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _create_checkpoint_manager(self):
        """Create checkpoint manager."""
        checkpoint_dir = self.working_dir + "/checkpoints"
        return CheckpointManager(self.logger, checkpoint_dir=checkpoint_dir)

    def _create_environment(self):
        """Create training environment."""
        return get_env(self.config.task, self.logger, self.config.seed)

    def _get_environment_specifications(self):
        """Get observation and action specifications from environment."""
        return get_env_specs(self.env, self.config.use_pixels)

    def _create_agent(self):
        """Create DecQN agent."""
        obs_shape, action_spec_dict = self._get_environment_specifications()
        return DecQNAgent(self.config, obs_shape, action_spec_dict)

    def train(self):
        """Execute main training loop."""
        self._setup_training()
        start_episode = self._load_checkpoint()
        metrics_tracker = self._initialize_metrics_tracker(start_episode)
        self._log_setup_info()
        self._run_training_loop(start_episode, metrics_tracker)
        self._finalize_training(metrics_tracker)
        return self.agent

    def _setup_training(self):
        """Initialize training environment."""
        init_training(self.config.seed, self.device, self.logger)

    def _load_checkpoint(self):
        """Load checkpoint if available."""
        return self.checkpoint_manager.load_checkpoint_if_available(
            self.config.load_checkpoints, self.agent
        )

    def _initialize_metrics_tracker(self, start_episode):
        """Initialize or load metrics tracker."""
        save_dir = self.working_dir + "/metrics"
        metrics_tracker = MetricsTracker(self.logger, save_dir)

        if start_episode > 0:
            metrics_tracker.load_metrics(self.config.load_metrics)

        return metrics_tracker

    def _log_setup_info(self):
        """Log training setup information."""
        self.logger.info("Decoupled Q-Networks Agent Setup:")
        self.logger.info(f"  Task: {self.config.task}")
        self.logger.info(f"  Decouple: {self.agent.config.decouple}")
        self.logger.info(f"  Action dimensions: {self.agent.action_discretizer.action_dim}")

    def _run_training_loop(self, start_episode, metrics_tracker):
        """Execute the main training loop."""
        metrics_accumulator = MetricsAccumulator()
        start_time = time.time()

        for episode in range(start_episode, self.config.num_episodes):
            episode_metrics = self._run_episode(metrics_accumulator)
            self._log_episode_metrics(episode, episode_metrics, start_time)
            _update_agent_parameters(self.agent)
            self._perform_periodic_maintenance(episode)
            self._save_checkpoint_if_needed(metrics_tracker, episode)
            metrics_tracker.log_episode(episode=episode, **episode_metrics)

    def _reset_episode(self):
        """Reset environment and prepare for new episode."""
        time_step = self.env.reset()
        obs = process_observation(
            time_step.observation, self.config.use_pixels, self.device
        )
        self.agent.observe_first(obs)
        return obs, time_step

    def _should_continue_episode(self, time_step, steps):
        """Check if episode should continue."""
        return not time_step.last() and steps < 1000

    def _execute_action(self, obs):
        """Select and execute action in environment."""
        action = self.agent.select_action(obs)
        action_np = _convert_action_to_numpy(action)
        time_step = self.env.step(action_np)
        return action, time_step

    def _process_transition(self, time_step):
        """Process environment transition."""
        next_obs = process_observation(
            time_step.observation, self.config.use_pixels, self.device
        )
        reward = time_step.reward if time_step.reward is not None else 0.0
        done = time_step.last()
        return next_obs, reward, done

    def _store_transition(self, action, reward, next_obs, done):
        """Store transition in agent's replay buffer."""
        self.agent.observe(action, reward, next_obs, done)

    def _collect_episode_metrics(self, episode_reward, steps, episode_time,
                                 metrics_accumulator):
        """Collect metrics for completed episode."""
        averages = metrics_accumulator.get_averages()
        return {
            "reward": episode_reward,
            "steps": steps,
            "loss": averages["loss"],
            "mean_abs_td_error": averages["mean_abs_td_error"],
            "mean_squared_td_error": averages["mean_squared_td_error"],
            "q_mean": averages["q_mean"],
            "epsilon": self.agent.epsilon,
            "mse_loss": averages["mse_loss"],
            "episode_time": episode_time,
        }

    def _run_episode(self, metrics_accumulator):
        """Run a single training episode."""
        episode_start_time = time.time()
        episode_reward = 0.0
        steps = 0

        obs, time_step = self._reset_episode()

        while self._should_continue_episode(time_step, steps):
            action, time_step = self._execute_action(obs)
            next_obs, reward, done = self._process_transition(time_step)

            self._store_transition(action, reward, next_obs, done)
            self._update_networks_if_ready(metrics_accumulator)

            obs = next_obs
            episode_reward += reward
            steps += 1

        episode_time = time.time() - episode_start_time
        return self._collect_episode_metrics(
            episode_reward, steps, episode_time, metrics_accumulator
        )

    def _has_sufficient_samples(self):
        """Check if replay buffer has enough samples."""
        return len(self.agent.replay_buffer) > self.config.min_replay_size

    def _update_networks_if_ready(self, metrics_accumulator):
        """Update networks if replay buffer has enough samples."""
        if not self._has_sufficient_samples():
            return

        metrics = self.agent.update()
        metrics_accumulator.update(metrics)

    def _should_log_basic(self, episode):
        """Check if basic metrics should be logged."""
        return episode % self.config.log_interval == 0

    def _should_log_detailed(self, episode):
        """Check if detailed metrics should be logged."""
        return episode % self.config.detailed_log_interval == 0 and episode > 0

    def _log_episode_metrics(self, episode, metrics, start_time):
        """Log episode metrics at specified intervals."""
        if self._should_log_basic(episode):
            self._log_basic_metrics(episode, metrics)

        if self._should_log_detailed(episode):
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
            f"Buffer: {len(self.agent.replay_buffer):6d}"
        )

    def _calculate_time_statistics(self, episode, start_time):
        """Calculate elapsed and estimated time."""
        elapsed_time = time.time() - start_time
        episodes_completed = episode + 1
        avg_episode_time = elapsed_time / episodes_completed
        remaining_episodes = self.config.num_episodes - episode - 1
        eta = avg_episode_time * remaining_episodes
        return elapsed_time, eta

    def _log_detailed_metrics(self, episode, start_time):
        """Log detailed training progress."""
        elapsed_time, eta = self._calculate_time_statistics(episode, start_time)

        self.logger.info(f"Episode {episode} Summary:")
        self.logger.info(
            f"Elapsed: {elapsed_time / 60:.1f}min | ETA: {eta / 60:.1f}min"
        )

    def _should_perform_gpu_cleanup(self, episode):
        """Check if GPU cleanup should be performed."""
        return episode % 10 == 0 and self.device == "cuda"

    def _cleanup_gpu_memory(self):
        """Clean up GPU memory."""
        torch.cuda.empty_cache()
        gc.collect()

    def _synchronize_gpu(self):
        """Synchronize GPU operations."""
        if self.device == "cuda":
            torch.cuda.synchronize()

    def _perform_periodic_maintenance(self, episode):
        """Perform periodic memory cleanup."""
        if self._should_perform_gpu_cleanup(episode):
            self._cleanup_gpu_memory()
        self._synchronize_gpu()

    def _should_save_checkpoint(self, episode):
        """Check if checkpoint should be saved."""
        return episode % self.config.checkpoint_interval == 0

    def _save_metrics(self, metrics_tracker):
        """Save current metrics."""
        metrics_tracker.save_metrics(
            self.agent_name, self.config.task, self.config.seed
        )

    def _save_agent_checkpoint(self, episode):
        """Save agent checkpoint."""
        return self.checkpoint_manager.save_checkpoint(
            self.agent, episode, self.config.task, self.config.seed
        )

    def _save_checkpoint_if_needed(self, metrics_tracker, episode):
        """Save checkpoint at specified intervals."""
        if not self._should_save_checkpoint(episode):
            return

        self._save_metrics(metrics_tracker)
        checkpoint_path = self._save_agent_checkpoint(episode)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

    def _save_final_checkpoint(self):
        """Save final checkpoint."""
        task_name = self.config.task + "_final"
        return self.checkpoint_manager.save_checkpoint(
            self.agent, self.config.num_episodes, task_name, self.config.seed
        )

    def _finalize_training(self, metrics_tracker):
        """Finalize training by saving and plotting."""
        self._save_metrics(metrics_tracker)
        final_checkpoint = self._save_final_checkpoint()
        self.logger.info(f"Final checkpoint saved: {final_checkpoint}")
        self._generate_plots(metrics_tracker)

    def _create_plotter(self, metrics_tracker):
        """Create plotting utility."""
        plot_dir = self.working_dir + "/plots"
        return PlottingUtils(self.logger, metrics_tracker, plot_dir)

    def _generate_plots(self, metrics_tracker):
        """Generate training plots."""
        self.logger.info("Generating plots...")
        plotter = self._create_plotter(metrics_tracker)
        plotter.plot_training_curves(save=True)
        plotter.plot_reward_distribution(save=True)
        plotter.print_summary_stats()


if __name__ == "__main__":
    args = parse_args()
    config = create_config_from_args(args)
    trainer = DecQNTrainer(config)
    trained_agent = trainer.train()