import gc
import time
import torch
import sys

sys.path.append('src')
from src.common.checkpoint_manager import CheckpointManager
from src.common.logger import Logger
from src.common.metrics_accumulator import MetricsAccumulator
from src.common.metrics_tracker import MetricsTracker
from src.common.training_utils import process_observation, get_env_specs, get_env, init_training
from src.plotting.plotting_utils import PlottingUtils
from src.gqn.agent import GQNAgent
from src.gqn.config import parse_args, create_config_from_args


def _convert_action_to_numpy(action):
    """Convert action tensor to numpy array."""
    if isinstance(action, torch.Tensor):
        return action.cpu().numpy()
    return action


def _create_episode_metrics_dict(episode_reward, steps, episode_time, averages, agent, success=None):
    """Create dictionary of episode metrics."""
    return {
        "reward": episode_reward,
        "steps": steps,
        "loss": averages.get("loss", 0.0),
        "mean_abs_td_error": averages.get("mean_abs_td_error", 0.0),
        "mean_squared_td_error": averages.get("mean_squared_td_error", 0.0),
        "q_mean": averages.get("q_mean", 0.0),
        "epsilon": agent.epsilon,
        "mse_loss": averages.get("mse_loss", 0.0),
        "episode_time": episode_time,
        "current_bins": agent.action_space_manager.current_bins,
        "growth_stage": agent.action_space_manager.current_growth_stage,
        "success": success,
    }


def _update_metrics_accumulator(metrics, metrics_accumulator):
    """Update metrics accumulator with default values."""
    metrics['loss'] = metrics.get('loss', 0.0)
    metrics['q1_mean'] = metrics.get('q1_mean', 0.0)
    metrics['mse_loss1'] = metrics.get('mse_loss1', 0.0)
    metrics['mean_abs_td_error'] = metrics.get('mean_abs_td_error', 0.0)
    metrics['mean_squared_td_error'] = metrics.get('mean_squared_td_error', 0.0)
    metrics_accumulator.update(metrics)


class GQNTrainer(Logger):
    """Trainer for Growing Q-Networks Agent managing the complete training pipeline."""

    def __init__(self, config, working_dir="./output/gqn"):
        super().__init__(working_dir + "/logs")
        self.working_dir = working_dir
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.agent_name = "gqn"
        self.checkpoint_manager = CheckpointManager(
            self.logger,
            checkpoint_dir=self.working_dir + "/checkpoints"
        )

    def train(self):
        """Execute main training loop."""
        self._setup_training()

        env = get_env(self.config.task, self.logger, self.config.seed, self.config.env_type)
        obs_shape, action_spec_dict = get_env_specs(env, self.config.use_pixels, self.config.env_type)

        agent = GQNAgent(self.config, obs_shape, action_spec_dict)

        start_episode = self.checkpoint_manager.load_checkpoint_if_available(
            self.config.load_checkpoints, agent
        )

        metrics_tracker = self._initialize_metrics_tracker(
            start_episode,
            save_dir=self.working_dir + "/metrics"
        )

        self._log_setup_info(agent)
        self._run_training_loop(env, agent, metrics_tracker, start_episode)
        self._finalize_training(agent, metrics_tracker)

        return agent

    def _setup_training(self):
        """Initialize training environment."""
        init_training(self.config.seed, self.device, self.logger)

    def _log_setup_info(self, agent):
        """Log training setup information."""
        self._print_separator()
        self.logger.info("Growing Q-Networks Agent Setup:")
        self._print_separator()

        self._log_environment_info()
        self._log_action_space_info(agent)
        self._log_learning_parameters()
        self._log_replay_buffer_info()
        self._log_regularization_info()
        self._log_logging_intervals()

        self._print_separator()

    def _print_separator(self):
        """Print a separator line."""
        self.logger.info("=" * 80)

    def _print_subseparator(self):
        """Print a subsection separator line."""
        self.logger.info("-" * 80)

    def _log_environment_info(self):
        """Log environment configuration."""
        self.logger.info(f"Environment Type: {self.config.env_type}")
        self.logger.info(f"Task: {self.config.task}")
        self.logger.info(f"Seed: {self.config.seed}")
        self.logger.info(f"Episodes: {self.config.num_episodes}")
        self.logger.info(f"Max steps per episode: {self.config.max_steps_per_episode}")
        self._print_subseparator()

    def _log_action_space_info(self, agent):
        """Log action space configuration."""
        self.logger.info(f"Action dimensions: {agent.action_space_manager.action_dim}")
        self.logger.info(f"Growth sequence: {agent.action_space_manager.growth_sequence}")
        self.logger.info(f"Growth schedule: {self.config.growing_schedule}")
        self.logger.info(f"Initial bins: {self.config.initial_bins}")
        self.logger.info(f"Final bins: {self.config.final_bins}")
        self._print_subseparator()

    def _log_learning_parameters(self):
        """Log learning hyperparameters."""
        self.logger.info(f"Learning rate: {self.config.learning_rate}")
        self.logger.info(f"Batch size: {self.config.batch_size}")
        self.logger.info(f"Discount: {self.config.discount}")
        self.logger.info(f"N-step: {self.config.n_step}")
        self.logger.info(f"Target update period: {self.config.target_update_period}")
        self.logger.info(
            f"Epsilon: {self.config.epsilon} (decay: {self.config.epsilon_decay}, min: {self.config.min_epsilon})")
        self._print_subseparator()

    def _log_replay_buffer_info(self):
        """Log replay buffer configuration."""
        self.logger.info(f"Min replay size: {self.config.min_replay_size}")
        self.logger.info(f"Max replay size: {self.config.max_replay_size}")
        self.logger.info(f"PER alpha: {self.config.per_alpha}")
        self.logger.info(f"PER beta: {self.config.per_beta}")
        self._print_subseparator()

    def _log_regularization_info(self):
        """Log regularization parameters."""
        self.logger.info(f"Action penalty coeff: {self.config.action_penalty_coeff}")
        self.logger.info(f"Gradient clip: {self.config.gradient_clip}")
        self.logger.info(f"Huber delta: {self.config.huber_delta}")
        self._print_subseparator()

    def _log_logging_intervals(self):
        """Log logging and checkpointing intervals."""
        self.logger.info(f"Checkpoint interval: {self.config.checkpoint_interval}")
        self.logger.info(f"Metrics save interval: {self.config.metrics_save_interval}")
        self.logger.info(f"Log interval: {self.config.log_interval}")
        self.logger.info(f"Detailed log interval: {self.config.detailed_log_interval}")

    def _run_training_loop(self, env, agent, metrics_tracker, start_episode):
        """Execute the main training loop."""
        self.agent = agent
        metrics_accumulator = MetricsAccumulator()
        start_time = time.time()

        for episode in range(start_episode, self.config.num_episodes):
            episode_metrics = self._run_episode(env, agent, metrics_accumulator)

            self._log_episode_metrics(episode, episode_metrics, start_time)
            self._check_and_log_growth(episode, episode_metrics, agent)

            agent.update_epsilon()

            self._perform_periodic_maintenance(episode)
            self._save_checkpoint_if_needed(agent, episode)
            self._save_metrics_if_needed(metrics_tracker, episode)

            metrics_tracker.log_episode(episode=episode, **episode_metrics)

    def _check_and_log_growth(self, episode, episode_metrics, agent):
        """Check if action space grew and log if it did."""
        grew = agent.check_and_grow(episode, episode_metrics['reward'])
        if grew:
            self._log_action_space_growth(episode, agent)

    def _log_action_space_growth(self, episode, agent):
        """Log action space growth event."""
        growth_info = agent.action_space_manager.get_growth_info()
        self.logger.info(
            f">>> ACTION SPACE GREW to {growth_info['current_bins']} bins "
            f"at episode {episode} (stage {growth_info['stage']}/{growth_info['total_stages'] - 1})"
        )

    def _run_episode(self, env, agent, metrics_accumulator):
        """Run a single training episode."""
        episode_start_time = time.time()
        episode_reward, steps, success = self._execute_episode_steps(env, agent, metrics_accumulator)
        episode_time = time.time() - episode_start_time
        averages = metrics_accumulator.get_averages()

        return _create_episode_metrics_dict(
            episode_reward, steps, episode_time, averages, agent
        )

    def _execute_episode_steps(self, env, agent, metrics_accumulator):
        """Execute all steps in an episode."""
        episode_reward = 0.0
        steps = 0
        success = None

        obs = self._reset_environment(env, agent)

        while steps < self.config.max_steps_per_episode:
            action = agent.select_action(obs)
            obs, reward, done, info = self._take_environment_step(env, agent, action)
            self._update_networks_if_ready(agent, metrics_accumulator)

            episode_reward += reward
            steps += 1
            if self.config.env_type == "metaworld" and info is not None:
                success = info.get('success', success)
            if done:
                break
        return episode_reward, steps, success

    def _reset_environment(self, env, agent):
        """Reset environment and initialize agent observation."""
        time_step = env.reset()
        obs = process_observation(
            time_step.observation,
            self.config.use_pixels,
            self.device,
            env_type=self.config.env_type
        )
        agent.observe_first(obs)
        return obs

    def _is_episode_complete(self, env, steps):
        """Check if episode is complete."""
        if self.config.env_type in ["ogbench", "metaworld"]:
            return steps >= self.config.max_steps_per_episode
        else:
            return env._step_count >= self.config.max_steps_per_episode or steps >= self.config.max_steps_per_episode

    def _take_environment_step(self, env, agent, action):
        """Take one step in the environment."""
        action_np = _convert_action_to_numpy(action)
        time_step = env.step(action_np)

        next_obs = process_observation(
            time_step.observation,
            self.config.use_pixels,
            self.device,
            env_type=self.config.env_type
        )
        reward = time_step.reward
        done = time_step.last()
        info = getattr(time_step, 'info', None)
        agent.observe(action, reward, next_obs, done)
        return next_obs, reward, done, info

    def _update_networks_if_ready(self, agent, metrics_accumulator):
        """Update networks if replay buffer has enough samples."""
        if len(agent.replay_buffer) <= self.config.min_replay_size:
            return

        metrics = agent.update()
        if metrics:
            _update_metrics_accumulator(metrics, metrics_accumulator)

    def _log_episode_metrics(self, episode, metrics, start_time):
        """Log episode metrics at specified intervals."""
        if episode % self.config.log_interval == 0:
            self._log_basic_metrics(episode, metrics)

        if episode % self.config.detailed_log_interval == 0 and episode > 0:
            self._log_detailed_metrics(episode, start_time)

    def _log_basic_metrics(self, episode, metrics):
        """Log basic episode metrics."""
        buffer_size = len(self.agent.replay_buffer) if hasattr(self, 'agent') else 0

        log_msg = (
            f"Ep {episode:4d} | "
            f"Steps {metrics['steps']:4d} | "
            f"R: {metrics['reward']:7.2f} | "
            f"Loss: {metrics['loss']:8.6f} | "
            f"MSE: {metrics['mse_loss']:8.6f} | "
            f"TD: {metrics['mean_abs_td_error']:8.6f} | "
            f"Q: {metrics['q_mean']:6.3f} | "
            f"Îµ: {metrics['epsilon']:.4f} | "
            f"Bins: {metrics.get('current_bins', 'N/A')} | "
            f"Time: {metrics['episode_time']:.2f}s | "
            f"Buf: {buffer_size:6d}"
            f"| Success: {metrics['success'] if metrics['success'] else 'N/A'}"
        )
        self.logger.info(log_msg)

    def _log_detailed_metrics(self, episode, start_time):
        """Log detailed training progress."""
        self._print_separator()
        self.logger.info(f"Episode {episode} Detailed Summary:")

        self._log_time_statistics(episode, start_time)
        self._log_agent_statistics(episode)

        self._print_separator()

    def _log_time_statistics(self, episode, start_time):
        """Log time-related statistics."""
        elapsed_time = time.time() - start_time
        episodes_completed = episode + 1
        avg_episode_time = elapsed_time / episodes_completed
        remaining_episodes = self.config.num_episodes - episode - 1
        eta = avg_episode_time * remaining_episodes

        self.logger.info(f"  Elapsed time: {elapsed_time / 60:.1f} min")
        self.logger.info(f"  Estimated time remaining: {eta / 60:.1f} min")
        self.logger.info(f"  Average episode time: {avg_episode_time:.2f} sec")
        self.logger.info(f"  Episodes completed: {episodes_completed}/{self.config.num_episodes}")
        self.logger.info(f"  Progress: {100 * episodes_completed / self.config.num_episodes:.1f}%")

    def _log_agent_statistics(self, episode):
        """Log agent-specific statistics."""
        if not hasattr(self, 'agent'):
            return

        growth_info = self.agent.action_space_manager.get_growth_info()
        self.logger.info(f"  Current bins: {growth_info['current_bins']}")
        self.logger.info(f"  Growth stage: {growth_info['stage']}/{growth_info['total_stages'] - 1}")
        self.logger.info(f"  Epsilon: {self.agent.epsilon:.4f}")
        self.logger.info(f"  Replay buffer size: {len(self.agent.replay_buffer)}")

    def _perform_periodic_maintenance(self, episode):
        """Perform periodic memory cleanup."""
        if episode % 10 == 0 and self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

        if self.device == "cuda":
            torch.cuda.synchronize()

    def _save_checkpoint_if_needed(self, agent, episode):
        """Save checkpoint at specified intervals."""
        if episode == 0 or (episode + 1) % self.config.checkpoint_interval != 0:
            return

        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            agent, episode, self.config.task, self.config.seed
        )
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

    def _save_metrics_if_needed(self, metrics_tracker, episode):
        """Save metrics at specified intervals."""
        if episode == 0 or (episode + 1) % self.config.metrics_save_interval != 0:
            return

        metrics_tracker.save_metrics(
            self.agent_name,
            self.config.task,
            self.config.seed,
            self.config.env_type
        )
        self.logger.info(f"Metrics saved at episode {episode}")

    def _finalize_training(self, agent, metrics_tracker):
        """Finalize training by saving and plotting."""
        self._save_final_metrics(metrics_tracker)
        self._save_final_checkpoint(agent)
        self._generate_plots(metrics_tracker)

    def _save_final_metrics(self, metrics_tracker):
        """Save final metrics."""
        metrics_tracker.save_metrics(
            self.agent_name,
            self.config.task,
            self.config.seed,
            self.config.env_type
        )

    def _save_final_checkpoint(self, agent):
        """Save final checkpoint."""
        final_checkpoint = self.checkpoint_manager.save_checkpoint(
            agent,
            self.config.num_episodes,
            self.config.task + "_final",
            self.config.seed
        )
        self.logger.info(f"Final checkpoint saved: {final_checkpoint}")

    def _generate_plots(self, metrics_tracker):
        """Generate training plots."""
        self.logger.info("Generating plots...")
        plotter = PlottingUtils(
            self.logger,
            metrics_tracker,
            self.working_dir + "/plots"
        )
        plotter.plot_training_curves(save=True)
        plotter.plot_reward_distribution(save=True)
        plotter.print_summary_stats()

    def _initialize_metrics_tracker(self, start_episode, save_dir):
        """Initialize or load metrics tracker."""
        metrics_tracker = MetricsTracker(self.logger, save_dir)

        if start_episode > 0 and self.config.load_metrics:
            metrics_tracker.load_metrics(self.config.load_metrics)

        return metrics_tracker


if __name__ == "__main__":
    args = parse_args()
    config = create_config_from_args(args)
    trainer = GQNTrainer(config)
    trained_agent = trainer.train()