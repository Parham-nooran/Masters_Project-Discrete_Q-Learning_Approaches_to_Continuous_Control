"""
Training loop for Coarse-to-Fine Q-Network agent.
"""

import argparse
import time
from typing import Dict

import numpy as np
import torch

from src.common.checkpoint_manager import CheckpointManager
from src.common.device_utils import get_device
from src.common.logger import Logger
from src.common.metrics_tracker import MetricsTracker
from src.common.training_utils import (
    get_env_specs,
    get_env,
    init_training,
)
from src.cqn.agent import CQNAgent
from src.cqn.config import CQNConfig, create_config


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


def _run_eval_episode(env, agent: CQNAgent, step: int) -> float:
    """
    Execute single evaluation episode.

    Args:
        env: Environment for evaluation.
        agent: CQN agent instance.
        step: Current training step.

    Returns:
        Episode cumulative reward.
    """
    time_step = env.reset()
    obs = _extract_observation(time_step)
    episode_reward = 0.0

    with torch.no_grad():
        while not time_step.last():
            action = agent.act(obs, step, eval_mode=True)
            time_step = env.step(action)
            obs = _extract_observation(time_step)
            episode_reward += time_step.reward

    return episode_reward


class ReplayBufferIterator:
    """
    Iterator wrapper for replay buffer.

    Provides continuous sampling from replay buffer.
    """

    def __init__(self, replay_buffer, batch_size: int):
        """
        Initialize iterator.

        Args:
            replay_buffer: Replay buffer to sample from.
            batch_size: Batch size for sampling.
        """
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        """Sample next batch from replay buffer."""
        return self.replay_buffer.sample(self.batch_size)


class CQNTrainer(Logger):
    """
    Trainer for Coarse-to-Fine Q-Network agent.

    Manages training loop, evaluation, checkpointing, and metrics tracking.
    """

    def __init__(self, config: CQNConfig, working_dir="./src/cqn/output"):
        """
        Initialize trainer.

        Args:
            config: CQNConfig object with hyperparameters.
            working_dir: Directory for logs and checkpoints.
        """
        super().__init__(working_dir + "/logs")
        self.working_dir = working_dir
        self.config = config
        self.device, _, _ = get_device()
        self.agent_name = "cqn"
        self.checkpoint_manager = CheckpointManager(
            self.logger, checkpoint_dir=self.working_dir + "/checkpoints"
        )

    def train(self) -> CQNAgent:
        """
        Execute main training loop.

        Returns:
            Trained agent instance.
        """
        self._setup_environment()
        env = get_env(self.config.task, self.logger)
        obs_shape, action_spec = get_env_specs(env)
        agent = CQNAgent(self.config, obs_shape, action_spec)

        start_episode = self.checkpoint_manager.load_checkpoint_if_available(
            self.config.load_checkpoints, agent
        )
        metrics_tracker = MetricsTracker(
            self.logger, save_dir=self.working_dir + "/metrics"
        )

        self._log_training_setup(agent)

        try:
            self._run_training_loop(env, agent, metrics_tracker, start_episode)
            self._finalize_training(metrics_tracker)
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
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

    def _log_training_setup(self, agent: CQNAgent) -> None:
        """Log training configuration."""
        self.logger.info("CQN Training Setup:")
        self.logger.info(f"  Task: {self.config.task}")
        self.logger.info(f"  Device: {self.device}")
        self.logger.info(f"  Num levels: {agent.num_levels}")
        self.logger.info(f"  Num bins: {agent.num_bins}")
        self.logger.info(f"  Num atoms: {self.config.num_atoms}")
        self.logger.info(f"  Learning rate: {self.config.lr}")

    def _run_training_loop(
            self,
            env,
            agent: CQNAgent,
            metrics_tracker: MetricsTracker,
            start_episode: int
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
        global_step = 0

        replay_iter = ReplayBufferIterator(agent.replay_buffer, self.config.batch_size)

        for episode in range(start_episode, self.config.max_episodes):
            episode_metrics, steps = self._run_episode(env, agent, replay_iter, global_step)
            global_step += steps

            self._log_episode_progress(episode, episode_metrics, start_time, global_step)

            if episode % self.config.eval_frequency == 0 and episode > 0:
                self._run_evaluation(env, agent, episode, global_step)

            if episode % self.config.save_frequency == 0 and episode > 0:
                self._save_checkpoint(agent, metrics_tracker, episode)

            metrics_tracker.log_episode(episode=episode, **episode_metrics)

    def _run_episode(
            self,
            env,
            agent: CQNAgent,
            replay_iter,
            global_step: int
    ) -> tuple[Dict[str, float], int]:
        """
        Execute single training episode.

        Args:
            env: Training environment.
            agent: CQN agent instance.
            replay_iter: Replay buffer iterator.
            global_step: Current global step count.

        Returns:
            Tuple of (episode metrics dict, number of steps).
        """
        episode_start_time = time.time()
        steps = 0
        time_step = env.reset()
        obs = _extract_observation(time_step)
        episode_reward = 0.0
        update_metrics = {}

        while not time_step.last():
            action = agent.act(obs, global_step + steps, eval_mode=False)
            time_step = env.step(action)
            next_obs = _extract_observation(time_step)
            reward = time_step.reward
            done = time_step.last()

            agent.store_transition(obs, action, reward, next_obs, done)

            if len(agent.replay_buffer) > self.config.min_buffer_size:
                update_metrics = agent.update(replay_iter, global_step + steps)

            obs = next_obs
            episode_reward += reward
            steps += 1

        episode_time = time.time() - episode_start_time

        metrics = {
            "reward": episode_reward,
            "steps": steps,
            "episode_time": episode_time,
            "buffer_size": len(agent.replay_buffer),
        }
        metrics.update(update_metrics)

        return metrics, steps

    def _log_episode_progress(
            self,
            episode: int,
            metrics: Dict[str, float],
            start_time: float,
            global_step: int
    ) -> None:
        """
        Log episode progress.

        Args:
            episode: Current episode number.
            metrics: Episode metrics.
            start_time: Training start time.
            global_step: Current global step.
        """
        if episode % self.config.log_interval == 0:
            elapsed = time.time() - start_time

            log_str = (
                f"Episode {episode}: "
                f"Reward={metrics['reward']:.2f}, "
                f"Steps={metrics['steps']}, "
                f"Time={metrics.get('episode_time', 0):.2f}s, "
                f"Elapsed={elapsed / 60:.1f}min, "
                f"GlobalStep={global_step}, "
                f"BufferSize={metrics.get('buffer_size', 0)}"
            )

            if 'critic_loss' in metrics:
                log_str += f", Loss={metrics['critic_loss']:.4f}"

            self.logger.info(log_str)

    def _run_evaluation(
            self,
            env,
            agent: CQNAgent,
            episode: int,
            global_step: int
    ) -> None:
        """
        Run evaluation episodes.

        Args:
            env: Environment for evaluation.
            agent: CQN agent instance.
            episode: Current episode number.
            global_step: Current global step.
        """
        eval_rewards = []
        num_eval = self.config.num_eval_episodes

        for _ in range(num_eval):
            eval_reward = _run_eval_episode(env, agent, global_step)
            eval_rewards.append(eval_reward)

        mean_reward = np.mean(eval_rewards)
        std_reward = np.std(eval_rewards)

        self.logger.info(
            f"Evaluation at episode {episode}: "
            f"Mean={mean_reward:.2f}, Std={std_reward:.2f}"
        )

    def _save_checkpoint(
            self,
            agent: CQNAgent,
            metrics_tracker: MetricsTracker,
            episode: int
    ) -> None:
        """
        Save training checkpoint.

        Args:
            agent: CQN agent instance.
            metrics_tracker: Metrics tracking object.
            episode: Episode number for checkpoint naming.
        """
        if episode % self.config.checkpoint_interval != 0:
            return
        metrics_tracker.save_metrics(self.agent_name, self.config.task, self.config.seed)
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            agent, episode, self.config.task, self.config.seed
        )
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

    def _finalize_training(
            self,
            metrics_tracker: MetricsTracker
    ) -> None:
        """
        Save final checkpoint and metrics.

        Args:
            agent: CQN agent instance.
            metrics_tracker: Metrics tracking object.
        """
        metrics_tracker.save_metrics(self.agent_name, self.config.task, self.config.seed)
        self.logger.info("Training completed")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train Coarse-to-Fine Q-Network agent"
    )

    parser.add_argument(
        "--task",
        type=str,
        default="walker_walk",
        help="DMControl task (format: domain_task)",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=1000,
        help="Maximum training episodes"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size"
    )
    parser.add_argument(
        "--num-levels",
        type=int,
        default=3,
        help="Hierarchy levels"
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=5,
        help="Bins per level"
    )
    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=50,
        help="Evaluation frequency (episodes)",
    )
    parser.add_argument(
        "--save-frequency",
        type=int,
        default=100,
        help="Checkpoint save frequency (episodes)",
    )
    parser.add_argument(
        "--working-dir",
        type=str,
        default=None,
        help="Working directory"
    )
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
        "--load-checkpoints",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    config = create_config(args)

    trainer = CQNTrainer(config)
    agent = trainer.train()
