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


class GQNTrainer(Logger):
    """Trainer for Growing Q-Networks Agent."""
    
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
        
        env = get_env(self.config.task, self.logger)
        obs_shape, action_spec_dict = get_env_specs(env, self.config.use_pixels)
        
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
        self.logger.info("Growing Q-Networks Agent Setup:")
        self.logger.info(f"  Task: {self.config.task}")
        self.logger.info(f"  Action dimensions: {agent.action_space_manager.action_dim}")
        self.logger.info(f"  Growth sequence: {agent.action_space_manager.growth_sequence}")
        self.logger.info(f"  Growth schedule: {self.config.growing_schedule}")
        self.logger.info(f"  Action penalty coeff: {self.config.action_penalty_coeff}")
    
    def _run_training_loop(self, env, agent, metrics_tracker, start_episode):
        """Execute the main training loop."""
        metrics_accumulator = MetricsAccumulator()
        start_time = time.time()
        
        for episode in range(start_episode, self.config.num_episodes):
            episode_metrics = self._run_episode(env, agent, metrics_accumulator)
            
            self._log_episode_metrics(episode, episode_metrics, start_time)
            
            grew = agent.check_and_grow(episode, episode_metrics['reward'])
            if grew:
                growth_info = agent.action_space_manager.get_growth_info()
                self.logger.info(
                    f"Action space grew to {growth_info['current_bins']} bins "
                    f"at episode {episode}"
                )
            
            agent.update_epsilon(decay_rate=0.995, min_epsilon=0.01)
            
            self._perform_periodic_maintenance(episode)
            self._save_checkpoint_if_needed(agent, metrics_tracker, episode)
            
            episode_metrics['current_bins'] = agent.action_space_manager.current_bins
            episode_metrics['growth_stage'] = agent.action_space_manager.current_growth_stage
            
            metrics_tracker.log_episode(episode=episode, **episode_metrics)
    
    def _run_episode(self, env, agent, metrics_accumulator):
        """Run a single training episode."""
        episode_start_time = time.time()
        episode_reward = 0.0
        steps = 0
        
        time_step = env.reset()
        obs = process_observation(
            time_step.observation, 
            self.config.use_pixels, 
            self.device
        )
        agent.observe_first(obs)
        
        while not time_step.last() and steps < self.config.max_steps_per_episode:
            action = agent.select_action(obs)
            action_np = self._convert_action_to_numpy(action)
            
            time_step = env.step(action_np)
            next_obs = process_observation(
                time_step.observation, 
                self.config.use_pixels, 
                self.device
            )
            reward = time_step.reward if time_step.reward is not None else 0.0
            done = time_step.last()
            
            agent.observe(action, reward, next_obs, done)
            
            self._update_networks_if_ready(agent, metrics_accumulator)
            
            obs = next_obs
            episode_reward += reward
            steps += 1
        
        episode_time = time.time() - episode_start_time
        averages = metrics_accumulator.get_averages()
        
        return {
            "reward": episode_reward,
            "steps": steps,
            "loss": averages["loss"],
            "mean_abs_td_error": averages["mean_abs_td_error"],
            "mean_squared_td_error": averages["mean_squared_td_error"],
            "q_mean": averages["q_mean"],
            "epsilon": agent.epsilon,
            "mse_loss": averages["mse_loss"],
            "episode_time": episode_time,
        }
    
    def _convert_action_to_numpy(self, action):
        """Convert action tensor to numpy array."""
        if isinstance(action, torch.Tensor):
            return action.cpu().numpy()
        return action
    
    def _update_networks_if_ready(self, agent, metrics_accumulator):
        """Update networks if replay buffer has enough samples."""
        if len(agent.replay_buffer) <= self.config.min_replay_size:
            return
        
        metrics = agent.update()
        if metrics:
            metrics_accumulator.update(metrics)
    
    def _log_episode_metrics(self, episode, metrics, start_time):
        """Log episode metrics at specified intervals."""
        if episode % self.config.log_interval == 0:
            self._log_basic_metrics(episode, metrics)
        
        if episode % self.config.detailed_log_interval == 0 and episode > 0:
            self._log_detailed_metrics(episode, start_time)
    
    def _log_basic_metrics(self, episode, metrics):
        """Log basic episode metrics."""
        self.logger.info(
            f"Episode {episode:4d} | "
            f"Steps {metrics['steps']:4d} | "
            f"Reward: {metrics['reward']:7.2f} | "
            f"Loss: {metrics['loss']:8.6f} | "
            f"MSE: {metrics['mse_loss']:8.6f} | "
            f"TD Error: {metrics['mean_abs_td_error']:8.6f} | "
            f"Q-mean: {metrics['q_mean']:6.3f} | "
            f"Epsilon: {metrics['epsilon']:.4f} | "
            f"Bins: {metrics.get('current_bins', 'N/A')} | "
            f"Time: {metrics['episode_time']:.2f}s"
        )
    
    def _log_detailed_metrics(self, episode, start_time):
        """Log detailed training progress."""
        elapsed_time = time.time() - start_time
        episodes_completed = episode + 1
        avg_episode_time = elapsed_time / episodes_completed
        remaining_episodes = self.config.num_episodes - episode - 1
        eta = avg_episode_time * remaining_episodes
        
        self.logger.info(f"Episode {episode} Summary:")
        self.logger.info(
            f"Elapsed: {elapsed_time / 60:.1f}min | ETA: {eta / 60:.1f}min"
        )
    
    def _perform_periodic_maintenance(self, episode):
        """Perform periodic memory cleanup."""
        if episode % 10 == 0 and self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
        
        if self.device == "cuda":
            torch.cuda.synchronize()
    
    def _save_checkpoint_if_needed(self, agent, metrics_tracker, episode):
        """Save checkpoint at specified intervals."""
        if episode % self.config.checkpoint_interval != 0:
            return
        
        metrics_tracker.save_metrics(
            self.agent_name, 
            self.config.task, 
            self.config.seed
        )
        
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            agent, episode, self.config.task, self.config.seed
        )
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _finalize_training(self, agent, metrics_tracker):
        """Finalize training by saving and plotting."""
        metrics_tracker.save_metrics(
            self.agent_name, 
            self.config.task, 
            self.config.seed
        )
        
        final_checkpoint = self.checkpoint_manager.save_checkpoint(
            agent, 
            self.config.num_episodes, 
            self.config.task + "_final", 
            self.config.seed
        )
        self.logger.info(f"Final checkpoint saved: {final_checkpoint}")
        
        self._generate_plots(metrics_tracker)
    
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