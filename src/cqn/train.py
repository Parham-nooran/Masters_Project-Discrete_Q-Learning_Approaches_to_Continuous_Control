"""
Training loop for Coarse-to-Fine Q-Network agent.
"""

import argparse
import time
from typing import Dict

import numpy as np
import torch

from src.common.checkpoint_manager import CheckpointManager
from src.common.logger import Logger
from src.common.metrics_tracker import MetricsTracker
from src.common.training_utils import (
    get_env_specs,
    get_env,
    init_training,
)
from src.cqn.agent import CQNAgent
from src.cqn.config import CQNConfig


def _extract_observation(time_step) -> np.ndarray:
    """
    Extract and flatten observation from time step.

    Args:
        time_step: DM Control time step.

    Returns:
        Flattened observation array.
    """
    obs_list = []
    for key in sorted(time_step.observation.keys()):
        obs_value = time_step.observation[key]
        if hasattr(obs_value, "flatten"):
            obs_list.append(obs_value.flatten())
        else:
            obs_list.append(np.array([obs_value]))

    return np.concatenate(obs_list)


class CQNTrainer(Logger):
    """
    Trainer for Coarse-to-Fine Q-Network agent.

    Manages training loop, evaluation, checkpointing, and metrics tracking.
    """

    def __init__(self, config: CQNConfig, working_dir: str ="./src/cqn/output/logs"):
        """
        Initialize trainer.

        Args:
            config: CQNConfig object with hyperparameters.
            working_dir: Directory for logs and checkpoints.
        """
        super().__init__(working_dir)
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_manager = CheckpointManager()

    def train(self) -> CQNAgent:
        """
        Execute main training loop.

        Returns:
            Trained agent instance.
        """
        self._setup_environment()

        env = get_env(self.config.task, self.logger)
        obs_shape, action_spec = get_env_specs(env)
        agent = CQNAgent(self.config, obs_shape, action_spec, working_dir="./src/cqn/output/")

        start_episode = self._load_checkpoint_if_available(agent)
        metrics_tracker = MetricsTracker(self.logger, save_dir="./src/cqn/output/logs")

        self._log_training_setup(agent)

        try:
            self._run_training_loop(env, agent, metrics_tracker, start_episode)
            self._finalize_training(agent, metrics_tracker)
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
            self._save_interrupted_checkpoint(agent, metrics_tracker)
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
        finally:
            env.close()
            self.logger.info("Environment closed")

        return agent

    def _setup_environment(self) -> None:
        """Initialize training environment and paths."""
        init_training(self.config.seed, self.device, self.logger)

    def _load_checkpoint_if_available(self, agent: CQNAgent) -> int:
        """
        Load checkpoint if specified or find latest.

        Args:
            agent: Agent to load checkpoint into.

        Returns:
            Starting episode number.
        """
        if self.config.load_checkpoints:
            return self.checkpoint_manager.load_checkpoint(
                agent, self.config.load_checkpoints, self.logger
            )

        latest = self.checkpoint_manager.find_latest_checkpoint()
        if latest:
            self.logger.info(f"Found latest checkpoint: {latest}")
            return self.checkpoint_manager.load_checkpoint(agent, latest, self.logger)

        return 0

    def _log_training_setup(self, agent: CQNAgent) -> None:
        """Log training configuration."""
        self.logger.info("CQN Training Setup:")
        self.logger.info(f"  Task: {self.config.task}")
        self.logger.info(f"  Device: {self.device}")
        self.logger.info(f"  Num levels: {agent.num_levels}")
        self.logger.info(f"  Num bins: {agent.num_bins}")

    def _run_training_loop(
        self, env, agent: CQNAgent, metrics_tracker: MetricsTracker, start_episode: int
    ) -> None:
        """
        Execute training loop for specified episodes.

        Args:
            env: Training environment.
            agent: CQN agent instance.
            metrics_tracker: Metrics tracking object.
            start_episode: Starting episode number.
        """
        start_time = time.time()

        for episode in range(start_episode, self.config.max_episodes):
            episode_metrics = self._run_episode(env, agent)

            self._log_episode_progress(episode, episode_metrics, start_time)

            if episode % self.config.eval_frequency == 0 and episode > 0:
                self._run_evaluation(env, agent, episode)

            if episode % self.config.save_frequency == 0 and episode > 0:
                self._save_checkpoint(agent, metrics_tracker, episode)

            metrics_tracker.log_episode(episode=episode, **episode_metrics)

    def _run_episode(self, env, agent: CQNAgent) -> Dict[str, float]:
        """
        Execute single training episode.

        Args:
            env: Training environment.
            agent: CQN agent instance.

        Returns:
            Episode metrics dictionary.
        """
        steps = 0
        time_step = env.reset()
        obs = _extract_observation(time_step)
        episode_rewards = []
        update_metrics = {}

        while not time_step.last():
            action = agent.select_action(torch.from_numpy(obs).float())
            time_step = env.step(action.numpy())
            next_obs = _extract_observation(time_step)
            reward = time_step.reward
            done = time_step.last()

            agent.store_transition(obs, action.numpy(), reward, next_obs, done)

            if len(agent.replay_buffer) > self.config.min_buffer_size:
                update_metrics = agent.update(self.config.batch_size)

            obs = next_obs
            episode_rewards.append(reward)
            steps += 1

        metrics = {
            "rewards": episode_rewards,
            "steps": steps,
        }
        metrics.update(update_metrics)

        return metrics

    def _log_episode_progress(
        self, episode: int, metrics: Dict[str, float], start_time: float
    ) -> None:
        """
        Log episode progress and metrics.

        Args:
            episode: Episode number.
            metrics: Episode metrics dictionary.
            start_time: Training start time.
        """
        if episode % 10 == 0:
            elapsed = time.time() - start_time
            self.logger.info(
                f"Episode {episode}: Reward={torch.sum(torch.tensor(metrics['rewards'])):.2f}, "
                f"Steps={metrics['steps']}, "
                f"Elapsed={elapsed / 60:.1f}min"
            )

    def _run_evaluation(self, env, agent: CQNAgent, episode: int) -> None:
        """
        Run evaluation episodes.

        Args:
            env: Environment for evaluation.
            agent: CQN agent instance.
            episode: Current episode number.
        """
        eval_rewards = []
        num_eval = 5

        for _ in range(num_eval):
            eval_reward = self._run_eval_episode(env, agent)
            eval_rewards.append(eval_reward)

        mean_reward = np.mean(eval_rewards)
        self.logger.info(f"Evaluation at episode {episode}: {mean_reward:.2f}")

    def _run_eval_episode(self, env, agent: CQNAgent) -> float:
        """
        Execute single evaluation episode.

        Args:
            env: Environment for evaluation.
            agent: CQN agent instance.

        Returns:
            Episode cumulative reward.
        """
        time_step = env.reset()
        obs = _extract_observation(time_step)
        episode_reward = 0.0

        with torch.no_grad():
            while not time_step.last():
                action = agent.select_action(
                    torch.from_numpy(obs).float(), evaluate=True
                )
                time_step = env.step(action.numpy())
                obs = _extract_observation(time_step)
                episode_reward += time_step.reward

        return episode_reward

    def _save_checkpoint(
        self, agent: CQNAgent, metrics_tracker: MetricsTracker, episode: int
    ) -> None:
        """
        Save training checkpoint.

        Args:
            agent: CQN agent instance.
            metrics_tracker: Metrics tracking object.
            episode: Episode number for checkpoint naming.
        """
        metrics_tracker.save_metrics()
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            agent, episode, f"cqn_episode_{episode}"
        )
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

    def _save_interrupted_checkpoint(
        self, agent: CQNAgent, metrics_tracker: MetricsTracker
    ) -> None:
        """Save checkpoint when training is interrupted."""
        agent.save(f"{self.config.save_dir}/cqn_agent_interrupted.pth")
        metrics_tracker.save_metrics()

    def _finalize_training(
        self, agent: CQNAgent, metrics_tracker: MetricsTracker
    ) -> None:
        """Save final checkpoint and metrics."""
        agent.save(f"{self.config.save_dir}/cqn_agent_final.pth")
        metrics_tracker.save_metrics()
        self.logger.info("Training completed")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Train Coarse-to-Fine Q-Network agent")

    parser.add_argument(
        "--task",
        type=str,
        default="walker_walk",
        help="DMControl task (format: domain_task)",
    )
    parser.add_argument(
        "--max-episodes", type=int, default=1000, help="Maximum training episodes"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--num-levels", type=int, default=3, help="Hierarchy levels")
    parser.add_argument("--num-bins", type=int, default=5, help="Bins per level")
    parser.add_argument(
        "--eval-frequency", type=int, default=50, help="Evaluation frequency (episodes)"
    )
    parser.add_argument(
        "--save-frequency",
        type=int,
        default=100,
        help="Checkpoint save frequency (episodes)",
    )
    parser.add_argument(
        "--working-dir", type=str, default=".", help="Working directory"
    )
    parser.add_argument(
        "--load-checkpoints",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    return parser.parse_args()


def create_config(args: argparse.Namespace) -> CQNConfig:
    """
    Create CQNConfig from parsed arguments.

    Args:
        args: Parsed command line arguments.

    Returns:
        CQNConfig instance.
    """
    config = CQNConfig()

    if args.task:
        config.task = args.task
    if args.max_episodes:
        config.max_episodes = args.max_episodes
    if args.seed is not None:
        config.seed = args.seed
    if args.learning_rate:
        config.lr = args.learning_rate
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.num_levels:
        config.num_levels = args.num_levels
    if args.num_bins:
        config.num_bins = args.num_bins
    if args.eval_frequency:
        config.eval_frequency = args.eval_frequency
    if args.save_frequency:
        config.save_frequency = args.save_frequency
    if args.working_dir:
        config.working_dir = args.working_dir
    if args.load_checkpoints:
        config.load_checkpoints = args.load_checkpoints

    return config


if __name__ == "__main__":
    args = parse_arguments()
    config = create_config(args)

    trainer = CQNTrainer(config, working_dir=args.working_dir)
    agent = trainer.train()
