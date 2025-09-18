import logging
import os
import time
from collections import deque
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from dm_control import suite

from agent import BangBangAgent
from src.common.metrics_tracker import MetricsTracker
from src.deqn.networks import LayerNormMLP


class BernoulliPolicy(nn.Module):
    """Bernoulli policy for bang-bang control as described in the paper."""

    def __init__(self, input_size: int, action_dim: int, hidden_sizes: list = [512, 512]):
        super().__init__()
        self.action_dim = action_dim
        sizes = [input_size] + hidden_sizes + [action_dim]
        self.network = LayerNormMLP(sizes, activate_final=False)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns logits for Bernoulli distribution."""
        return self.network(obs)

    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from Bernoulli policy."""
        logits = self.forward(obs)
        probs = torch.sigmoid(logits)

        if deterministic:
            actions = (probs > 0.5).float()
        else:
            dist = torch.distributions.Bernoulli(probs)
            actions = dist.sample()

        # Convert to bang-bang actions: 0 -> -1, 1 -> +1
        bang_bang_actions = 2.0 * actions - 1.0
        log_probs = torch.distributions.Bernoulli(probs).log_prob(actions).sum(dim=-1)

        return bang_bang_actions, log_probs


class BangBangTrainer:
    """Trainer for Bang-Bang Control Agent."""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        self.logger.info(f"Using device: {self.device}")

    def process_observation(self, dm_obs: dict, use_pixels: bool) -> torch.Tensor:
        """Process DM Control observation."""
        if use_pixels:
            if "pixels" in dm_obs:
                obs = dm_obs["pixels"]
            else:
                camera_obs = [v for k, v in dm_obs.items() if "camera" in k or "rgb" in k]
                if camera_obs:
                    obs = camera_obs[0]
                else:
                    raise ValueError("No pixel observations found")

            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
            if len(obs.shape) == 3:  # HWC -> CHW
                obs = obs.permute(2, 0, 1)
        else:
            state_parts = []
            for key in sorted(dm_obs.keys()):
                val = dm_obs[key]
                if isinstance(val, np.ndarray):
                    state_parts.append(val.astype(np.float32).flatten())
                else:
                    state_parts.append(np.array([float(val)], dtype=np.float32))

            state_vector = np.concatenate(state_parts, dtype=np.float32)
            obs = torch.from_numpy(state_vector).to(self.device)

        return obs

    def get_obs_shape(self, use_pixels: bool, obs_spec: dict) -> tuple:
        """Get observation shape."""
        if use_pixels:
            return (3, 84, 84)
        else:
            state_dim = sum(
                spec.shape[0] if len(spec.shape) > 0 else 1
                for spec in obs_spec.values()
            )
            return (state_dim,)

    def train(self):
        """Main training loop."""
        # Environment setup
        if self.config.task not in [f"{domain}_{task}" for domain, task in suite.ALL_TASKS]:
            self.logger.warn(f"Task {self.config.task} not found, using walker_walk")
            self.config.task = "walker_walk"

        domain_name, task_name = self.config.task.split("_", 1)
        env = suite.load(domain_name, task_name)
        action_spec = env.action_spec()
        obs_spec = env.observation_spec()

        obs_shape = self.get_obs_shape(self.config.use_pixels, obs_spec)
        action_spec_dict = {"low": action_spec.minimum, "high": action_spec.maximum}

        agent = BangBangAgent(self.config, obs_shape, action_spec_dict)

        # Metrics tracking
        os.makedirs("output/checkpoints", exist_ok=True)
        metrics_tracker = MetricsTracker(save_dir="output/metrics")

        self.logger.info(f"Starting Bang-Bang training on {self.config.task}")
        self.logger.info(f"Action dimension: {agent.action_dim}")

        start_time = time.time()

        for episode in range(self.config.num_episodes):
            episode_start_time = time.time()
            episode_reward = 0
            recent_losses = deque(maxlen=20)

            time_step = env.reset()
            obs = self.process_observation(time_step.observation, self.config.use_pixels)
            agent.observe_first(obs)

            step = 0
            while not time_step.last():
                action = agent.select_action(obs)
                action_np = action.cpu().numpy()

                time_step = env.step(action_np)
                next_obs = self.process_observation(
                    time_step.observation, self.config.use_pixels
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

                if step > 1000:  # Episode length limit
                    break

            # Log metrics
            avg_loss = np.mean(recent_losses) if recent_losses else 0.0

            metrics_tracker.log_episode(
                episode=episode,
                reward=episode_reward,
                length=step,
                loss=avg_loss,
                mean_abs_td_error=0.0,  # Not applicable for policy gradient
                mean_squared_td_error=0.0,
                q_mean=0.0,
                epsilon=0.0  # Not applicable
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
                recent_rewards = metrics_tracker.episode_rewards[-self.config.detailed_log_interval:]
                self.logger.info(f"Recent avg reward: {np.mean(recent_rewards):.2f}")
                self.logger.info(f"ETA: {eta / 60:.1f} min")

            # Checkpointing
            if episode % self.config.checkpoint_interval == 0:
                checkpoint_path = f"output/checkpoints/bangbang_{self.config.task}_{episode}.pth"
                agent.save_checkpoint(checkpoint_path, episode)
                metrics_tracker.save_metrics()

        # Final save
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

    # Bang-Bang specific defaults
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
    import argparse

    parser = argparse.ArgumentParser(description="Train Bang-Bang Control Agent")
    parser.add_argument("--task", type=str, default="walker_walk", help="Environment task")
    parser.add_argument("--num-episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--log-interval", type=int, default=10, help="Log interval")
    parser.add_argument("--detailed-log-interval", type=int, default=50, help="Detailed log interval")
    parser.add_argument("--checkpoint-interval", type=int, default=100, help="Checkpoint interval")

    args = parser.parse_args()
    config = create_bangbang_config(args)

    trainer = BangBangTrainer(config)
    agent = trainer.train()
