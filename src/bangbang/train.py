import argparse
import time
from collections import deque

from src.bangbang.agent import BangBangAgent
from src.bangbang.config import create_bangbang_config
from src.common.logger import Logger
from src.common.metrics_tracker import MetricsTracker
from src.common.training_utils import *


class BangBangTrainer(Logger):

    def __init__(self, config, working_dir="./src/bangbang/output"):
        super().__init__(working_dir + "/logs")
        self.working_dir = working_dir
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def train(self):
        init_training(self.config.seed, self.device, self.logger)
        env = get_env(self.config.task, self.logger)
        obs_shape, action_spec_dict = get_env_specs(env, self.config.use_pixels)
        agent = BangBangAgent(self.config, obs_shape, action_spec_dict)

        metrics_tracker = MetricsTracker(self.logger, save_dir="output/metrics")

        self._log_training_start(agent)

        start_time = time.time()

        for episode in range(self.config.num_episodes):
            episode_metrics = self._run_episode(env, agent, episode)

            metrics_tracker.log_episode(
                episode=episode,
                reward=episode_metrics["reward"],
                steps=episode_metrics["steps"],
                loss=episode_metrics["avg_loss"],
                mean_abs_td_error=0.0,
                mean_squared_td_error=0.0,
                q_mean=0.0,
                epsilon=0.0,
            )

            self._log_episode_progress(episode, episode_metrics, agent)
            self._log_detailed_progress(episode, metrics_tracker, start_time)
            self._save_checkpoint_if_needed(episode, agent, metrics_tracker)

        self._finalize_training(agent, metrics_tracker, start_time)

        return agent

    def _log_training_start(self, agent):
        self.logger.info(f"Starting Bang-Bang training on {self.config.task}")
        self.logger.info(f"Action dimension: {agent.action_dim}")

    def _run_episode(self, env, agent, episode):
        episode_start_time = time.time()
        episode_reward = 0
        recent_losses = deque(maxlen=20)

        time_step = env.reset()
        obs = process_observation(
            time_step.observation, self.config.use_pixels, self.device
        )
        agent.observe_first(obs)

        steps = 0
        while not time_step.last() and steps < 1000:
            action = agent.select_action(obs)

            time_step = self._execute_action(env, action)
            next_obs = process_observation(
                time_step.observation, self.config.use_pixels, self.device
            )
            reward = time_step.reward if time_step.reward is not None else 0.0
            done = time_step.last()

            agent.observe(action, reward, next_obs, done)

            if self._should_update(agent):
                metrics = agent.update()
                if metrics and "policy_loss" in metrics:
                    recent_losses.append(metrics["policy_loss"])

            obs = next_obs
            episode_reward += reward
            steps += 1

        return {
            "reward": episode_reward,
            "steps": steps,
            "avg_loss": self._compute_average_loss(recent_losses),
            "time": time.time() - episode_start_time,
        }

    def _execute_action(self, env, action):
        action_np = action.cpu().numpy()
        return env.step(action_np)

    def _should_update(self, agent):
        return len(agent.replay_buffer) > self.config.min_replay_size

    def _compute_average_loss(self, recent_losses):
        return np.mean(recent_losses) if recent_losses else 0.0

    def _log_episode_progress(self, episode, episode_metrics, agent):
        if episode % self.config.log_interval == 0:
            self.logger.info(
                f"Episode {episode:4d} | "
                f"Reward: {episode_metrics['reward']:7.2f} | "
                f"Loss: {episode_metrics['avg_loss']:8.6f} | "
                f"Time: {episode_metrics['time']:.2f}s | "
                f"Buffer: {len(agent.replay_buffer):6d}"
            )

    def _log_detailed_progress(self, episode, metrics_tracker, start_time):
        if episode % self.config.detailed_log_interval == 0 and episode > 0:
            elapsed_time = time.time() - start_time
            avg_episode_time = elapsed_time / episode
            eta = avg_episode_time * (self.config.num_episodes - episode)

            recent_rewards = metrics_tracker.episode_rewards[
                -self.config.detailed_log_interval :
            ]

            self.logger.info(f"Episode {episode} Summary:")
            self.logger.info(f"Recent avg reward: {np.mean(recent_rewards):.2f}")
            self.logger.info(f"ETA: {eta / 60:.1f} min")

    def _save_checkpoint_if_needed(self, episode, agent, metrics_tracker):
        if episode % self.config.checkpoint_interval == 0:
            checkpoint_path = (
                f"output/checkpoints/bangbang_{self.config.task}_{episode}.pth"
            )
            agent.save_checkpoint(checkpoint_path, episode)
            metrics_tracker.save_metrics()

    def _finalize_training(self, agent, metrics_tracker, start_time):
        final_path = f"output/checkpoints/bangbang_{self.config.task}_final.pth"
        agent.save_checkpoint(final_path, self.config.num_episodes)
        metrics_tracker.save_metrics()

        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time / 60:.1f} minutes!")


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
