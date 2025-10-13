import argparse
import time
from collections import deque

from src.bangbang.agent import BangBangAgent
from src.common.logger import Logger
from src.common.metrics_tracker import MetricsTracker
from src.common.training_utils import *


class BangBangTrainer(Logger):
    """Trainer for Bang-Bang Control Agent."""

    def __init__(self, config, working_dir="."):
        super().__init__(working_dir)
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def train(self):
        """Main training loop."""
        init_training(self.config.seed, self.device, self.logger)
        env = get_env(self.config.task, self.logger)
        obs_shape, action_spec_dict = get_env_specs(env, self.config.use_pixels)
        agent = BangBangAgent(self.config, obs_shape, action_spec_dict)

        metrics_tracker = MetricsTracker(save_dir="output/metrics")

        self.logger.info(f"Starting Bang-Bang training on {self.config.task}")
        self.logger.info(f"Action dimension: {agent.action_dim}")

        start_time = time.time()

        for episode in range(self.config.num_episodes):
            episode_start_time = time.time()
            episode_reward = 0
            recent_losses = deque(maxlen=20)

            time_step = env.reset()
            obs = process_observation(
                time_step.observation, self.config.use_pixels, self.device
            )
            agent.observe_first(obs)

            step = 0
            while not time_step.last():
                action = agent.select_action(obs)
                action_np = action.cpu().numpy()

                time_step = env.step(action_np)
                next_obs = process_observation(
                    time_step.observation, self.config.use_pixels, device=self.device
                )
                reward = time_step.reward if time_step.reward is not None else 0.0
                done = time_step.last()

                agent.observe(action, reward, next_obs, done)

                if len(agent.replay_buffer) > self.config.min_replay_size:
                    metrics = agent.update()
                    if metrics and "policy_loss" in metrics:
                        recent_losses.append(metrics["policy_loss"])

                obs = next_obs
                episode_reward += reward
                step += 1

                if step > 1000:
                    break

            avg_loss = np.mean(recent_losses) if recent_losses else 0.0

            metrics_tracker.log_episode(
                episode=episode,
                reward=episode_reward,
                length=step,
                loss=avg_loss,
                mean_abs_td_error=0.0,
                mean_squared_td_error=0.0,
                q_mean=0.0,
                epsilon=0.0,
            )

            episode_time = time.time() - episode_start_time

            if episode % self.config.log_interval == 0:
                self.logger.info(
                    f"Episode {episode:4d} | "
                    f"Reward: {episode_reward:7.2f} | "
                    f"Loss: {avg_loss:8.6f} | "
                    f"Time: {episode_time:.2f}s | "
                    f"Buffer: {len(agent.replay_buffer):6d}"
                )

            if episode % self.config.detailed_log_interval == 0 and episode > 0:
                elapsed_time = time.time() - start_time
                avg_episode_time = elapsed_time / episode
                eta = avg_episode_time * (self.config.num_episodes - episode)

                self.logger.info(f"Episode {episode} Summary:")
                self.logger.info(f"Reward: {episode_reward:.2f}")
                recent_rewards = metrics_tracker.episode_rewards[
                    -self.config.detailed_log_interval :
                ]
                self.logger.info(f"Recent avg reward: {np.mean(recent_rewards):.2f}")
                self.logger.info(f"ETA: {eta / 60:.1f} min")

            if episode % self.config.checkpoint_interval == 0:
                checkpoint_path = (
                    f"output/checkpoints/bangbang_{self.config.task}_{episode}.pth"
                )
                agent.save_checkpoint(checkpoint_path, episode)
                metrics_tracker.save_metrics()

        final_path = f"output/checkpoints/bangbang_{self.config.task}_final.pth"
        agent.save_checkpoint(final_path, self.config.num_episodes)
        metrics_tracker.save_metrics()

        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time / 60:.1f} minutes!")

        return agent


def create_bangbang_config(args):
    """Create config for Bang-Bang agent."""
    from types import SimpleNamespace

    config = SimpleNamespace()
    for key, value in vars(args).items():
        setattr(config, key.replace("-", "_"), value)

    config.use_pixels = False
    config.layer_size_network = [512, 512]
    config.layer_size_bottleneck = 100
    config.num_pixels = 84
    config.min_replay_size = 1000
    config.max_replay_size = 500000
    config.batch_size = 128
    config.learning_rate = 3e-4
    config.discount = 0.99
    config.priority_exponent = 0.6
    config.importance_sampling_exponent = 0.4
    config.adder_n_step = 1
    config.clip_gradients = True
    config.clip_gradients_norm = 40.0

    return config


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

    args = parser.parse_args()
    config = create_bangbang_config(args)

    trainer = BangBangTrainer(config)
    agent = trainer.train()
