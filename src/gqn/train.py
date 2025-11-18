"""Training script for Growing Q-Networks Agent."""

import argparse
import gc
import json
import os
import time
from collections import deque

import numpy as np
import torch

from src.common.logger import Logger
from src.common.metrics_tracker import MetricsTracker
from src.common.observation_utils import process_observation
from src.common.replay_buffer import OptimizedObsBuffer
from src.common.training_utils import get_env, get_env_specs, init_training
from src.gqn.agent import GrowingQNAgent
from src.gqn.config import GQNConfig
from src.plotting.plotting_utils import PlottingUtils
from src.common.checkpoint_manager import CheckpointManager


def _init_recent_metrics():
    """Initialize recent metrics buffers."""
    return {
        "losses": deque(maxlen=20),
        "q1_means": deque(maxlen=20),
        "mean_abs_td_errors": deque(maxlen=20),
        "mean_squared_td_errors": deque(maxlen=20),
    }


def _init_episode_metrics():
    """Initialize episode metrics dictionary."""
    return {
        "reward": 0.0,
        "action_magnitudes": [],
        "steps": 0,
        "update_metrics": [],
    }


def _accumulate_update_metrics(episode_metrics, update_metrics):
    """Accumulate metrics from update step."""
    if update_metrics and isinstance(update_metrics, dict):
        episode_metrics["update_metrics"].append(update_metrics)


def _update_recent_metrics(recent_metrics, episode_metrics):
    """Update recent metrics buffers."""
    for update_metric in episode_metrics["update_metrics"]:
        if "q1_mean" in update_metric:
            recent_metrics["q1_means"].append(update_metric["q1_mean"])
        if "mean_abs_td_error" in update_metric:
            recent_metrics["mean_abs_td_errors"].append(
                update_metric["mean_abs_td_error"]
            )
        if "mean_squared_td_error" in update_metric:
            recent_metrics["mean_squared_td_errors"].append(
                update_metric["mean_squared_td_error"]
            )
        if "loss" in update_metric and update_metric["loss"] is not None:
            if not np.isnan(update_metric["loss"]):
                recent_metrics["losses"].append(update_metric["loss"])


def _compute_average_metrics(recent_metrics, episode_metrics):
    """Compute average of recent metrics."""
    return {
        "loss": (
            np.mean(recent_metrics["losses"]) if recent_metrics["losses"] else 0.0
        ),
        "q_mean": (
            np.mean(recent_metrics["q1_means"]) if recent_metrics["q1_means"] else 0.0
        ),
        "mean_abs_td_error": (
            np.mean(recent_metrics["mean_abs_td_errors"])
            if recent_metrics["mean_abs_td_errors"]
            else 0.0
        ),
        "mean_squared_td_error": (
            np.mean(recent_metrics["mean_squared_td_errors"])
            if recent_metrics["mean_squared_td_errors"]
            else 0.0
        ),
        "action_magnitude": torch.mean(
            torch.tensor(episode_metrics["action_magnitudes"])
        )
        / max(episode_metrics["steps"], 1),
    }


def _update_metrics(
    metrics_tracker, recent_metrics, episode, episode_metrics, agent, episode_time
):
    """Update all metrics tracking."""
    _update_recent_metrics(recent_metrics, episode_metrics)
    avg_metrics = _compute_average_metrics(recent_metrics, episode_metrics)

    growth_info = agent.get_growth_info()
    metrics_tracker.log_episode(
        episode=episode,
        reward=episode_metrics["reward"],
        steps=episode_metrics["steps"],
        loss=avg_metrics["loss"],
        mean_abs_td_error=avg_metrics["mean_abs_td_error"],
        mean_squared_td_error=avg_metrics["mean_squared_td_error"],
        q_mean=avg_metrics["q_mean"],
        epsilon=agent.epsilon,
        episode_time=episode_time,
        current_bins=growth_info["current_bins"],
        growth_history=str(growth_info["growth_history"]),
    )


class GQNTrainer(Logger):
    """Trainer for Growing Q-Networks Agent."""

    def __init__(self, config, working_dir="./src/gqn/output"):
        super().__init__(working_dir + "/logs")
        self.working_dir = working_dir
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.checkpoint_manager = CheckpointManager(
            self.logger,
            os.path.join(self.working_dir, "checkpoints")
        )

    def train(self):
        """Main training loop."""
        agent, metrics_tracker, start_episode = self._setup_training()
        self._log_setup_info(agent, start_episode)

        recent_metrics = _init_recent_metrics()
        obs_buffer = OptimizedObsBuffer(agent.obs_shape, self.device)
        start_time = time.time()

        for episode in range(start_episode, self.config.num_episodes):
            episode_start_time = time.time()
            episode_metrics = self._run_episode(agent, obs_buffer)
            episode_time = time.time() - episode_start_time

            _update_metrics(
                metrics_tracker,
                recent_metrics,
                episode,
                episode_metrics,
                agent,
                episode_time,
            )
            agent.update_epsilon(decay_rate=0.995, min_epsilon=0.01)

            self._log_progress(
                episode,
                episode_metrics,
                recent_metrics,
                agent,
                start_time,
                start_episode,
            )
            self._periodic_maintenance(episode, metrics_tracker, agent)

        self._finalize_training(agent, metrics_tracker, start_time)
        return agent

    def _setup_training(self):
        """Setup training components."""
        init_training(self.config.seed, self.device, self.logger)
        env = get_env(self.config.task, self.logger)
        obs_shape, action_spec_dict = get_env_specs(env, self.config.use_pixels)

        agent = GrowingQNAgent(
            self.config, obs_shape, action_spec_dict, self.working_dir
        )
        metrics_tracker = MetricsTracker(self.logger, save_dir=self.working_dir + "/metrics")

        start_episode = self._load_checkpoint_if_exists(agent, metrics_tracker)
        return agent, metrics_tracker, start_episode

    def _load_checkpoint_if_exists(self, agent, metrics_tracker):
        """Load checkpoint if specified."""
        if not self.config.load_checkpoints:
            return 0
        start_episode = self.checkpoint_manager.load_checkpoint_if_available(
            self.config.load_checkpoints, agent
        )

        if start_episode > 0:
            metrics_tracker.load_metrics()

        return start_episode

    def _log_setup_info(self, agent, start_episode):
        """Log training setup information."""
        self.logger.info("Growing Q-Networks Agent Setup:")
        self.logger.info(f"  Task: {self.config.task}")
        self.logger.info(f"  Decouple: {agent.config.decouple}")
        self.logger.info(f"  Action dimensions: {agent.action_discretizer.action_dim}")
        self.logger.info(f"  Growing schedule: {self.config.growing_schedule}")
        self.logger.info(
            f"  Growth sequence: {agent.action_discretizer.growth_sequence}"
        )
        self.logger.info(f"  Current bins: {agent.action_discretizer.num_bins}")
        self.logger.info(f"  Action penalty: {self.config.action_penalty}")
        self.logger.info(f"Starting training from episode {start_episode}...")

    def _run_episode(self, agent, obs_buffer):
        """Run a single training episode."""
        env = get_env(self.config.task, self.logger)
        time_step = env.reset()
        obs = process_observation(
            time_step.observation, self.config.use_pixels, self.device, obs_buffer
        ).to(self.device, non_blocking=True)
        agent.observe_first(obs)

        episode_metrics = _init_episode_metrics()
        steps = 0

        while not time_step.last() and steps < 1000:
            action = agent.select_action(obs)
            action_np = action.cpu().numpy()

            time_step = env.step(action_np)
            next_obs = process_observation(
                time_step.observation, self.config.use_pixels, self.device, obs_buffer
            ).to(self.device, non_blocking=True)

            reward, original_reward = self._compute_reward(action_np, time_step)
            agent.observe(action, reward, next_obs, time_step.last())

            update_metrics = self._update_if_ready(agent)
            _accumulate_update_metrics(episode_metrics, update_metrics)

            obs = next_obs
            episode_metrics["reward"] += reward
            episode_metrics["action_magnitudes"].append(np.linalg.norm(action_np))
            steps += 1

        episode_metrics["steps"] = steps
        agent.end_episode(episode_metrics["reward"])
        self._cleanup_memory()

        return episode_metrics

    def _compute_reward(self, action_np, time_step):
        """Compute penalized and original reward."""
        original_reward = time_step.reward if time_step.reward is not None else 0.0

        if self.config.action_penalty > 0:
            penalty = self.config.action_penalty * np.sum(action_np**2) / len(action_np)
            penalty = min(penalty, abs(original_reward) * 0.1)
            reward = original_reward - penalty
        else:
            reward = original_reward

        return reward, original_reward

    def _update_if_ready(self, agent):
        """Update agent if replay buffer is ready."""
        if len(agent.replay_buffer) > self.config.min_replay_size:
            return agent.update()
        return {}

    def _cleanup_memory(self):
        """Clean up GPU memory."""
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def _log_progress(
        self, episode, episode_metrics, recent_metrics, agent, start_time, start_episode
    ):
        """Log training progress."""
        if episode % self.config.log_interval == 0:
            self._log_regular_progress(episode, episode_metrics, recent_metrics, agent)

        if episode % self.config.detailed_log_interval == 0 and episode > 0:
            self._log_detailed_progress(
                episode, episode_metrics, agent, start_time, start_episode
            )

    def _log_regular_progress(self, episode, episode_metrics, recent_metrics, agent):
        """Log regular progress information."""
        avg_metrics = _compute_average_metrics(recent_metrics, episode_metrics)
        growth_info = agent.get_growth_info()

        self.logger.info(
            f"Episode {episode:4d} | "
            f"Reward: {episode_metrics['reward']:7.2f} | "
            f"Loss: {avg_metrics['loss']:8.6f} | "
            f"Bins: {growth_info['current_bins']} | "
            f"Growth history: {growth_info['growth_history']}"
        )

    def _log_detailed_progress(
        self, episode, episode_metrics, agent, start_time, start_episode
    ):
        """Log detailed progress information."""
        elapsed_time = time.time() - start_time
        avg_episode_time = elapsed_time / (episode - start_episode + 1)
        eta = avg_episode_time * (self.config.num_episodes - episode - 1)

        growth_info = agent.get_growth_info()
        self.logger.info(f"Episode {episode} Detailed Summary:")
        self.logger.info(f"Penalized Reward: {episode_metrics['reward']:.2f}")
        self.logger.info(f"Current resolution: {growth_info['current_bins']} bins")
        self.logger.info(f"Growth history: {growth_info['growth_history']}")
        self.logger.info(f"Buffer size: {len(agent.replay_buffer)}")
        self.logger.info(
            f"Elapsed: {elapsed_time / 60:.1f}min | ETA: {eta / 60:.1f}min"
        )

    def _periodic_maintenance(self, episode, metrics_tracker, agent):
        """Perform periodic maintenance tasks."""
        if episode % 100 == 0 and self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

        if episode % self.config.checkpoint_interval == 0:
            metrics_tracker.save_metrics(agent, self.config.task)
            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                agent, episode, self.config.task
            )
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")

    def _finalize_training(self, agent, metrics_tracker, start_time):
        """Finalize training and save results."""
        total_time = time.time() - start_time

        if metrics_tracker.episode_times:
            avg_episode_time = np.mean(metrics_tracker.episode_times)
            total_episodes = len(metrics_tracker.episode_times)
        else:
            avg_episode_time = 0
            total_episodes = 0

        metrics_tracker.save_metrics(agent, self.config.task)
        final_checkpoint = self.checkpoint_manager.save_checkpoint(
            agent, self.config.num_episodes, f"{self.config.task}_final"
        )
        self.logger.info(f"Final checkpoint saved: {final_checkpoint}")

        self.logger.info(f"Training completed in {total_time / 60:.1f} minutes!")
        self.logger.info(f"Average episode time: {avg_episode_time:.2f} seconds")
        self.logger.info(f"Total episodes: {total_episodes}")

        self._log_growth_summary(agent)
        self._save_timing_data(metrics_tracker, total_time)
        self._generate_plots(metrics_tracker)

    def _log_growth_summary(self, agent):
        """Log growth summary."""
        growth_info = agent.get_growth_info()
        self.logger.info("Growing Q-Networks Summary:")
        self.logger.info(f"  Final resolution: {growth_info['current_bins']} bins")
        self.logger.info(f"  Growth sequence achieved: {growth_info['growth_history']}")
        self.logger.info(
            f"  Total resolution levels: {len(growth_info['growth_history'])}"
        )

    def _save_timing_data(self, metrics_tracker, total_time):
        """Save timing information."""
        growth_events = metrics_tracker.get_growth_events()

        avg_episode_time = (
            np.mean(metrics_tracker.episode_times)
            if metrics_tracker.episode_times
            else 0
        )

        timing_data = {
            "total_time_seconds": total_time,
            "total_time_minutes": total_time / 60,
            "total_time_hours": total_time / 3600,
            "avg_episode_time_seconds": avg_episode_time,
            "num_episodes": len(metrics_tracker.episode_times),
            "growth_events": growth_events,
            "final_bins": (
                metrics_tracker.episode_current_bins[-1]
                if metrics_tracker.episode_current_bins
                else None
            ),
        }

        os.makedirs("output/metrics", exist_ok=True)
        with open("output/metrics/timing_data.json", "w") as f:
            json.dump(timing_data, f, indent=2)

        self.logger.info(f"Timing data saved to output/metrics/timing_data.json")
        self.logger.info(f"Number of growth events: {len(growth_events)}")

    def _generate_plots(self, metrics_tracker):
        """Generate training plots."""
        self.logger.info("Generating plots...")
        plotter = PlottingUtils(self.logger, metrics_tracker, self.working_dir + "/plots")
        plotter.plot_training_curves(save=True)
        plotter.plot_reward_distribution(save=True)
        plotter.print_summary_stats()



def create_gqn_config(args):
    """Create config for GQN agent."""
    config = GQNConfig.get_default_gqn_config(args)
    config.action_penalty = getattr(args, "action_penalty", 0.001)
    config.learning_rate = getattr(args, "learning_rate", 1e-4)
    config.batch_size = getattr(args, "batch_size", 256)
    config.target_update_period = getattr(args, "target_update_period", 100)
    config.layer_size_network = [512, 512]
    return config


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Growing Q-Networks Agent")

    parser.add_argument(
        "--load-checkpoints",
        type=str,
        default=None,
        help="Path to checkpoint file to resume from",
    )
    parser.add_argument(
        "--task", type=str, default="walker_walk", help="Environment task"
    )
    parser.add_argument(
        "--num-episodes", type=int, default=1000, help="Number of episodes to train"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--max-bins", type=int, default=9, help="Maximum number of bins"
    )
    parser.add_argument(
        "--growing-schedule",
        type=str,
        default="adaptive",
        choices=["linear", "adaptive"],
        help="Growing schedule type",
    )
    parser.add_argument(
        "--action-penalty", type=float, default=0.1, help="Action penalty coefficient"
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
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100,
        help="Save checkpoints every N episodes",
    )
    parser.add_argument(
        "--log-interval", type=int, default=10, help="Log progress every N episodes"
    )
    parser.add_argument(
        "--detailed-log-interval",
        type=int,
        default=50,
        help="Detailed log every N episodes",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    config = create_gqn_config(args)
    trainer = GQNTrainer(config)
    agent = trainer.train()
