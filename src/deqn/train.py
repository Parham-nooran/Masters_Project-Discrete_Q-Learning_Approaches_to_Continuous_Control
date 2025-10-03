import argparse
import gc
import os
import time
from collections import deque

import numpy as np
import torch
from dm_control import suite

from src.common.logger import Logger
from src.common.metrics_tracker import MetricsTracker
from src.common.utils import process_observation, get_obs_shape
from src.deqn.agent import DecQNAgent
from src.deqn.config import create_config_from_args
from src.plotting.plotting_utils import PlottingUtils


class DecQNTrainer(Logger):
    """Trainer for Decoupled Q-Networks Agent."""

    def __init__(self, config, working_dir="."):
        super().__init__(working_dir)
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def save_checkpoint(self, agent, episode, path):
        """Save agent checkpoint."""
        checkpoint = {
            "episode": episode,
            "q_network_state_dict": agent.q_network.state_dict(),
            "target_q_network_state_dict": agent.target_q_network.state_dict(),
            "q_optimizer_state_dict": agent.q_optimizer.state_dict(),
            "config": agent.config,
            "training_step": agent.training_step,
            "epsilon": agent.epsilon,
            "replay_buffer_buffer": agent.replay_buffer.buffer,
            "replay_buffer_position": agent.replay_buffer.position,
            "replay_buffer_priorities": agent.replay_buffer.priorities,
            "replay_buffer_max_priority": agent.replay_buffer.max_priority,
        }

        if agent.encoder:
            checkpoint["encoder_state_dict"] = agent.encoder.state_dict()
            checkpoint["encoder_optimizer_state_dict"] = agent.encoder_optimizer.state_dict()

        torch.save(checkpoint, path)

    def find_latest_checkpoint(self, checkpoint_dir="output/checkpoints"):
        """Find the most recent checkpoint file."""
        if not os.path.exists(checkpoint_dir):
            return None

        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
        if not checkpoint_files:
            return None

        checkpoint_files.sort(
            key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True
        )
        latest_file = checkpoint_files[0]

        return os.path.join(checkpoint_dir, latest_file)

    def load_checkpoint(self, agent, checkpoint_path):
        """Load checkpoint and return starting episode."""
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                loaded_episode = agent.load_checkpoint(checkpoint_path)
                start_episode = loaded_episode + 1
                self.logger.info(f"Resumed from episode {loaded_episode}, starting at {start_episode}")
                return start_episode
            except Exception as e:
                self.logger.error(f"Failed to load checkpoint: {e}")
                self.logger.info("Starting fresh training...")
                return 0
        else:
            if checkpoint_path:
                self.logger.warn(f"Checkpoint {checkpoint_path} not found. Starting fresh...")
            return 0

    def train(self):
        """Main training loop."""
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        self.logger.info(f"Using device: {self.device}")

        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True

        if self.config.task not in [f"{domain}_{task}" for domain, task in suite.ALL_TASKS]:
            self.logger.warn(f"Task {self.config.task} not found, using walker_walk")
            self.config.task = "walker_walk"

        domain_name, task_name = self.config.task.split("_", 1)
        env = suite.load(domain_name, task_name)
        action_spec = env.action_spec()
        obs_spec = env.observation_spec()

        obs_shape = get_obs_shape(self.config.use_pixels, obs_spec)
        action_spec_dict = {"low": action_spec.minimum, "high": action_spec.maximum}

        agent = DecQNAgent(self.config, obs_shape, action_spec_dict)

        os.makedirs("output/checkpoints", exist_ok=True)
        os.makedirs("output/metrics", exist_ok=True)
        os.makedirs("output/plots", exist_ok=True)

        start_episode = 0
        if self.config.load_checkpoints:
            start_episode = self.load_checkpoint(agent, self.config.load_checkpoints)
        else:
            latest = self.find_latest_checkpoint()
            if latest:
                self.logger.info(f"Found latest checkpoint: {latest}")
                start_episode = self.load_checkpoint(agent, latest)

        metrics_tracker = MetricsTracker(save_dir="output/metrics")
        if start_episode > 0:
            metrics_tracker.load_metrics()

        self.logger.info("Decoupled Q-Networks Agent Setup:")
        self.logger.info(f"  Task: {self.config.task}")
        self.logger.info(f"  Decouple: {agent.config.decouple}")
        self.logger.info(f"  Action dimensions: {agent.action_discretizer.action_dim}")
        self.logger.info(f"Starting training from episode {start_episode}...")

        recent_losses = deque(maxlen=20)
        recent_q1_means = deque(maxlen=20)
        recent_mean_abs_td_errors = deque(maxlen=20)
        recent_squared_td_errors = deque(maxlen=20)
        recent_mse_losses = deque(maxlen=20)

        start_time = time.time()

        for episode in range(start_episode, self.config.num_episodes):
            episode_start_time = time.time()
            episode_reward = 0

            time_step = env.reset()
            obs = process_observation(time_step.observation, self.config.use_pixels, self.device)
            agent.observe_first(obs)

            step = 0
            while not time_step.last():
                action = agent.select_action(obs)
                action_np = action.cpu().numpy() if isinstance(action, torch.Tensor) else action

                time_step = env.step(action_np)
                next_obs = process_observation(
                    time_step.observation, self.config.use_pixels, self.device
                )
                reward = time_step.reward if time_step.reward is not None else 0.0
                done = time_step.last()

                agent.observe(action, reward, next_obs, done)

                if len(agent.replay_buffer) > self.config.min_replay_size:
                    metrics = agent.update()
                    if metrics:
                        recent_q1_means.append(metrics["q1_mean"])
                        recent_mse_losses.append(metrics["mse_loss1"])
                        recent_mean_abs_td_errors.append(metrics["mean_abs_td_error"])
                        recent_squared_td_errors.append(metrics["mean_squared_td_error"])
                        if "loss" in metrics and metrics["loss"] is not None:
                            recent_losses.append(metrics["loss"])

                obs = next_obs
                episode_reward += reward
                step += 1

                if step > 1000:
                    break

            avg_recent_loss = np.mean(recent_losses) if recent_losses else 0.0
            avg_recent_q_means = np.mean(recent_q1_means) if recent_q1_means else 0.0
            avg_recent_mean_td_error = (
                np.mean(recent_mean_abs_td_errors) if recent_mean_abs_td_errors else 0.0
            )
            avg_recent_squared_td_error = (
                np.mean(recent_squared_td_errors) if recent_squared_td_errors else 0.0
            )
            avg_recent_mse_loss = np.mean(recent_mse_losses) if recent_mse_losses else 0.0
            metrics_tracker.log_episode(
                episode=episode,
                reward=episode_reward,
                length=step,
                loss=avg_recent_loss if recent_losses else None,
                mean_abs_td_error=avg_recent_mean_td_error,
                mean_squared_td_error=avg_recent_squared_td_error,
                q_mean=avg_recent_q_means if recent_q1_means else None,
                epsilon=agent.epsilon,
                mse_loss=avg_recent_mse_loss if recent_mse_losses else None
            )

            agent.update_epsilon(decay_rate=0.995, min_epsilon=0.01)

            if self.device == "cuda":
                torch.cuda.synchronize()

            episode_time = time.time() - episode_start_time

            if episode % self.config.log_interval == 0:
                self.logger.info(
                    f"Episode {episode:4d} | "
                    f"Reward: {episode_reward:7.2f} | "
                    f"Loss: {avg_recent_loss:8.6f} | "
                    f"MSE Loss: {avg_recent_mse_loss:8.6f} | "
                    f"Mean abs TD Error: {avg_recent_mean_td_error:8.6f} | "
                    f"Mean squared TD Error: {avg_recent_squared_td_error:8.6f} | "
                    f"Q-mean: {avg_recent_q_means:6.3f} | "
                    f"Time: {episode_time:.2f}s | "
                    f"Buffer: {len(agent.replay_buffer):6d}"
                )

            if episode % self.config.detailed_log_interval == 0 and episode > 0:
                elapsed_time = time.time() - start_time
                avg_episode_time = elapsed_time / (episode - start_episode + 1)
                eta = avg_episode_time * (self.config.num_episodes - episode - 1)

                recent_rewards = metrics_tracker.episode_rewards[-self.config.detailed_log_interval:]

                self.logger.info(f"Episode {episode} Summary:")
                self.logger.info(f"Cumulative Reward: {episode_reward:.2f}")
                self.logger.info(
                    f"Recent {self.config.detailed_log_interval} episodes avg reward: {np.mean(recent_rewards):.2f}"
                )
                self.logger.info(f"Elapsed: {elapsed_time / 60:.1f}min | ETA: {eta / 60:.1f}min")

            if episode % 10 == 0 and self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()

            if episode % self.config.checkpoint_interval == 0:
                metrics_tracker.save_metrics()
                checkpoint_path = f"output/checkpoints/decqn_{self.config.task}_{episode}.pth"
                self.save_checkpoint(agent, episode, checkpoint_path)
                self.logger.info(f"Checkpoint saved: {checkpoint_path}")

        metrics_tracker.save_metrics()
        final_checkpoint = f"output/checkpoints/decqn_{self.config.task}_final.pth"
        self.save_checkpoint(agent, self.config.num_episodes, final_checkpoint)
        self.logger.info(f"Final checkpoint saved: {final_checkpoint}")

        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time / 60:.1f} minutes!")

        self.logger.info("Generating plots...")
        plotter = PlottingUtils(metrics_tracker, save_dir="output/plots")
        plotter.plot_training_curves(save=True)
        plotter.plot_reward_distribution(save=True)
        plotter.print_summary_stats()

        return agent


def create_decqn_config(args):
    """Create config for DecQN agent."""
    config = create_config_from_args(args)
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Decoupled Q-Networks Agent")
    parser.add_argument(
        "--load-checkpoints",
        type=str,
        default=None,
        help="Path to checkpoint file to resume from",
    )
    parser.add_argument("--task", type=str, default="walker_walk", help="Environment task")
    parser.add_argument(
        "--num-episodes", type=int, default=1000, help="Number of episodes to train"
    )
    parser.add_argument(
        "--num-bins", type=int, default=2, help="Number of bins"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument("--epsilon", type=float, default=0.1, help="Epsilon for exploration")
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

    args = parser.parse_args()
    config = create_decqn_config(args)

    trainer = DecQNTrainer(config)
    agent = trainer.train()