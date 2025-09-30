import argparse
import os

import gym
import numpy as np
import torch

from src.common.logger import Logger
from src.common.metrics_tracker import MetricsTracker
from agent import CQNAgent
from config import CQNConfig


class CQNTrainer(Logger):
    """Trainer for CQN agent that handles the training loop and evaluation."""

    def __init__(self, config, working_dir="."):
        super().__init__(
            working_dir=working_dir,
            specific_excel_file_name="cqn_training",
            log_level=config.log_level,
            enable_logging=True,
            log_to_file=True,
            log_to_console=True
        )
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create environment
        self.env = self.create_environment(config.env_name, config.seed)

        # Create agent
        obs_shape = self.env.observation_space.shape
        action_spec = {
            "low": self.env.action_space.low,
            "high": self.env.action_space.high
        }
        self.agent = CQNAgent(config, obs_shape, action_spec, self.logger)

        self.metrics_tracker = MetricsTracker(save_dir=config.save_dir)

    def create_environment(self, env_name, seed):
        """Create and configure environment."""
        env = gym.make(env_name)
        env.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        return env

    def evaluate(self, num_episodes=5):
        """Evaluate the agent."""
        total_reward = 0

        for _ in range(num_episodes):
            obs = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                with torch.no_grad():
                    action = self.agent.select_action(
                        torch.from_numpy(obs).float(),
                        evaluate=True
                    )
                obs, reward, done, _ = self.env.step(action.numpy())
                episode_reward += reward

            total_reward += episode_reward

        return total_reward / num_episodes

    def train(self):
        """Main training loop."""
        self.logger.info("Starting CQN training...")
        self.logger.info(f"Environment: {self.config.env_name}")
        self.logger.info(f"Observation space: {self.env.observation_space}")
        self.logger.info(f"Action space: {self.env.action_space}")

        episode = 0
        total_steps = 0

        try:
            while episode < self.config.max_episodes:
                obs = self.env.reset()
                episode_reward = 0
                episode_length = 0
                done = False
                metrics = {}

                while not done:
                    action = self.agent.select_action(torch.from_numpy(obs).float())
                    next_obs, reward, done, info = self.env.step(action.numpy())

                    self.agent.store_transition(obs, action.numpy(), reward, next_obs, done)

                    if len(self.agent.replay_buffer) > self.config.min_buffer_size:
                        metrics = self.agent.update(self.config.batch_size)

                        if metrics and total_steps % 1000 == 0:
                            self.logger.info(
                                f"Step {total_steps}, Episode {episode}: "
                                f"Loss: {metrics.get('loss', 0.0):.4f}, "
                                f"Epsilon: {metrics.get('epsilon', 0.0):.4f}, "
                                f"Q-mean: {metrics.get('q_mean', 0.0):.4f}"
                            )

                    obs = next_obs
                    episode_reward += reward
                    episode_length += 1
                    total_steps += 1

                # Log episode metrics
                log_dict = {
                    "episode": episode,
                    "reward": episode_reward,
                    "length": episode_length,
                }
                if metrics:
                    log_dict.update(metrics)

                self.metrics_tracker.log_episode(**log_dict)

                self.logger.info(
                    f"Episode {episode}: Reward: {episode_reward:.2f}, "
                    f"Length: {episode_length}, Steps: {total_steps}"
                )

                # Evaluation
                if episode % self.config.eval_frequency == 0 and episode > 0:
                    eval_reward = self.evaluate()
                    self.logger.info(f"Evaluation at episode {episode}: {eval_reward:.2f}")

                # Save checkpoint
                if episode % self.config.save_frequency == 0 and episode > 0:
                    save_path = f"{self.config.save_dir}/cqn_agent_episode_{episode}.pth"
                    self.agent.save(save_path)
                    self.metrics_tracker.save_metrics()
                    self.logger.info(f"Checkpoint saved: {save_path}")

                episode += 1

            # Final save
            self.agent.save(f"{self.config.save_dir}/cqn_agent_final.pth")
            self.metrics_tracker.save_metrics()
            self.logger.info("Training completed!")

        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
            self.agent.save(f"{self.config.save_dir}/cqn_agent_interrupted.pth")
            self.metrics_tracker.save_metrics()
        except Exception as e:
            self.logger.error(f"Training failed with error: {e}")
            raise
        finally:
            self.env.close()
            self.logger.info("Environment closed")

        return self.agent


def create_cqn_config(args):
    """Create config for CQN agent from arguments."""
    config = CQNConfig()

    # Override with command line arguments
    if hasattr(args, 'env_name') and args.env_name:
        config.env_name = args.env_name
    if hasattr(args, 'max_episodes') and args.max_episodes:
        config.max_episodes = args.max_episodes
    if hasattr(args, 'seed') and args.seed is not None:
        config.seed = args.seed
    if hasattr(args, 'learning_rate') and args.learning_rate:
        config.learning_rate = args.learning_rate
    if hasattr(args, 'batch_size') and args.batch_size:
        config.batch_size = args.batch_size
    if hasattr(args, 'eval_frequency') and args.eval_frequency:
        config.eval_frequency = args.eval_frequency
    if hasattr(args, 'save_frequency') and args.save_frequency:
        config.save_frequency = args.save_frequency
    if hasattr(args, 'log_level') and args.log_level:
        config.log_level = args.log_level

    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CQN Agent")
    parser.add_argument(
        "--env-name",
        type=str,
        default="HalfCheetah-v2",
        help="Gym environment name"
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=1000,
        help="Maximum number of episodes"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate"
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=50,
        help="Evaluation frequency (episodes)"
    )
    parser.add_argument(
        "--save-frequency",
        type=int,
        default=100,
        help="Checkpoint save frequency (episodes)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--working-dir",
        type=str,
        default=".",
        help="Working directory for logs"
    )

    args = parser.parse_args()
    config = create_cqn_config(args)

    trainer = CQNTrainer(config, working_dir=args.working_dir)
    agent = trainer.train()