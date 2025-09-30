import gym
import numpy as np
import torch

from agent import CQNAgent
from config import CQNConfig
from src.common.logger import Logger


class CQNTrainer:
    """
    Trainer for CQN agent that handles the training loop and evaluation
    """

    def __init__(self, config, env, logger: Logger):
        self.config = config
        self.env = env
        self.logger = logger
        obs_shape = env.observation_space.shape
        action_spec = {
            "low": env.action_space.low,
            "high": env.action_space.high
        }
        self.agent = CQNAgent(config, obs_shape, action_spec, logger)
        self.max_episodes = config.max_episodes
        self.eval_frequency = config.eval_frequency
        self.save_frequency = config.save_frequency

    def train(self):
        """Main training loop"""
        self.logger.logger.info("Starting CQN training...")

        episode = 0
        total_steps = 0

        while episode < self.max_episodes:
            obs = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            while not done:
                action = self.agent.select_action(torch.from_numpy(obs).float())
                next_obs, reward, done, info = self.env.step(action.numpy())
                self.agent.store_transition(obs, action.numpy(), reward, next_obs, done)
                if len(self.agent.replay_buffer) > self.config.min_buffer_size:
                    metrics = self.agent.update(self.config.batch_size)

                    if metrics:
                        if total_steps % 1000 == 0:
                            self.logger.logger.info(
                                f"Step {total_steps}, Episode {episode}: "
                                f"Loss: {metrics['loss']:.4f}, "
                                f"Epsilon: {metrics['epsilon']:.4f}, "
                                f"Q-mean: {metrics['q_mean']:.4f}"
                            )

                obs = next_obs
                episode_reward += reward
                episode_length += 1
                total_steps += 1

            self.agent.metrics_tracker.log_episode(
                episode=episode,
                reward=episode_reward,
                length=episode_length,
                **metrics if 'metrics' in locals() and metrics else {}
            )

            self.logger.logger.info(
                f"Episode {episode}: Reward: {episode_reward:.2f}, "
                f"Length: {episode_length}, Steps: {total_steps}"
            )
            if episode % self.eval_frequency == 0:
                eval_reward = self.evaluate()
                self.logger.logger.info(f"Evaluation at episode {episode}: {eval_reward:.2f}")
            if episode % self.save_frequency == 0:
                save_path = f"{self.config.save_dir}/cqn_agent_episode_{episode}.pth"
                self.agent.save(save_path)

            episode += 1

        self.agent.save(f"{self.config.save_dir}/cqn_agent_final.pth")
        self.agent.metrics_tracker.save_metrics()
        self.logger.logger.info("Training completed!")

    def evaluate(self, num_episodes: int = 5) -> float:
        """Evaluate the agent"""
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


def create_environment(env_name: str, seed: int):
    """Create and configure environment"""
    env = gym.make(env_name)
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    return env


def main():
    """Main training function"""
    config = CQNConfig()
    logger = Logger(
        working_dir=config.working_dir,
        specific_excel_file_name="cqn_training",
        log_level=getattr(__import__('logging'), config.log_level),
        enable_logging=True,
        log_to_file=True,
        log_to_console=True
    )

    logger.logger.info("Starting CQN experiment")
    logger.logger.info(f"Configuration: {config}")

    env = create_environment(config.env_name, config.seed)
    logger.logger.info(f"Environment: {config.env_name}")
    logger.logger.info(f"Observation space: {env.observation_space}")
    logger.logger.info(f"Action space: {env.action_space}")
    trainer = CQNTrainer(config, env, logger)
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.logger.info("Training interrupted by user")
        trainer.agent.save(f"{config.save_dir}/cqn_agent_interrupted.pth")
        trainer.agent.metrics_tracker.save_metrics()
    except Exception as e:
        logger.logger.error(f"Training failed with error: {e}")
        raise
    finally:
        env.close()
        logger.logger.info("Environment closed")


if __name__ == "__main__":
    main()
