import time
import gc
from dm_control import suite
import os
import argparse
from config import GQNConfig

from agent import GrowingQNAgent
from plotting.plotting_utils import MetricsTracker, PlottingUtils
from train_utils import *


def parse_gqn_args():
    """Parse arguments specific to Growing Q-Networks."""
    parser = argparse.ArgumentParser(description="Train Growing Q-Networks Agent")

    # Base training parameters
    parser.add_argument("--load-checkpoints", type=str, default=None, help="Path to checkpoint file to resume from")
    parser.add_argument("--task", type=str, default="walker_walk", help="Environment task")
    parser.add_argument("--num-episodes", type=int, default=1000, help="Number of episodes to train")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    # GQN specific parameters
    parser.add_argument("--max-bins", type=int, default=9, help="Maximum number of bins (final resolution)")
    parser.add_argument("--growing-schedule", type=str, default="adaptive", choices=["linear", "adaptive"],
                        help="Growing schedule type")
    parser.add_argument("--action-penalty", type=float, default=0.1, help="Action penalty coefficient (ca)")

    # Standard RL parameters
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Epsilon for exploration")
    parser.add_argument("--discount", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--target-update-period", type=int, default=100, help="Target network update period")
    parser.add_argument("--min-replay-size", type=int, default=1000, help="Minimum replay buffer size")
    parser.add_argument("--max-replay-size", type=int, default=500000, help="Maximum replay buffer size")

    # Training control
    parser.add_argument("--checkpoint-interval", type=int, default=100, help="Save checkpoints every N episodes")
    parser.add_argument("--log-interval", type=int, default=10, help="Log progress every N episodes")
    parser.add_argument("--detailed-log-interval", type=int, default=50, help="Detailed log every N episodes")

    return parser.parse_args()


def apply_action_penalty(reward, action, penalty_coeff):
    """Apply action penalty as used in paper experiments."""
    if penalty_coeff > 0:
        if isinstance(action, torch.Tensor):
            action_np = action.cpu().numpy()
        else:
            action_np = action

        # Penalty: -ca * ||a||^2 / M (as in paper)
        action_penalty = penalty_coeff * np.sum(action_np ** 2) / len(action_np)
        return reward - action_penalty
    return reward


def save_checkpoint(agent, episode, path, metrics_tracker):
    """Save agent checkpoint including GQN-specific state."""
    checkpoint = {
        "episode": episode,
        "q_network_state_dict": agent.q_network.state_dict(),
        "target_q_network_state_dict": agent.target_q_network.state_dict(),
        "q_optimizer_state_dict": agent.q_optimizer.state_dict(),
        "config": agent.config,
        "training_step": agent.training_step,
        "epsilon": agent.epsilon,
        "episode_count": agent.episode_count,

        # GQN specific state
        "current_resolution_level": agent.current_resolution_level,
        "growth_history": agent.growth_history,
        "action_discretizer_current_bins": agent.action_discretizer.current_bins,
        "action_discretizer_current_growth_idx": agent.action_discretizer.current_growth_idx,

        # Replay buffer state
        "replay_buffer_buffer": agent.replay_buffer.buffer,
        "replay_buffer_position": agent.replay_buffer.position,
        "replay_buffer_priorities": agent.replay_buffer.priorities,
        "replay_buffer_max_priority": agent.replay_buffer.max_priority,

        # Scheduler state
        "scheduler_returns_history": list(agent.scheduler.returns_history),
        "scheduler_last_growth_episode": agent.scheduler.last_growth_episode,
    }

    if agent.encoder:
        checkpoint["encoder_state_dict"] = agent.encoder.state_dict()
        checkpoint["encoder_optimizer_state_dict"] = agent.encoder_optimizer.state_dict()

    torch.save(checkpoint, path)


def load_checkpoint(agent, checkpoint_path):
    """Load checkpoint and restore GQN-specific state."""
    if not os.path.exists(checkpoint_path):
        return 0

    try:
        checkpoint = torch.load(checkpoint_path, map_location=agent.device, weights_only=False)

        # Load standard network states
        agent.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        agent.target_q_network.load_state_dict(checkpoint["target_q_network_state_dict"])
        agent.q_optimizer.load_state_dict(checkpoint["q_optimizer_state_dict"])
        agent.training_step = checkpoint.get("training_step", 0)
        agent.epsilon = checkpoint.get("epsilon", agent.config.epsilon)
        agent.episode_count = checkpoint.get("episode_count", 0)

        # Load GQN specific state
        if "current_resolution_level" in checkpoint:
            agent.current_resolution_level = checkpoint["current_resolution_level"]
        if "growth_history" in checkpoint:
            agent.growth_history = checkpoint["growth_history"]
        if "action_discretizer_current_bins" in checkpoint:
            agent.action_discretizer.current_bins = checkpoint["action_discretizer_current_bins"]
        if "action_discretizer_current_growth_idx" in checkpoint:
            agent.action_discretizer.current_growth_idx = checkpoint["action_discretizer_current_growth_idx"]
            # Update action bins
            agent.action_discretizer.action_bins = agent.action_discretizer.all_action_bins[
                agent.action_discretizer.current_bins
            ]

        # Load replay buffer state
        if "replay_buffer_buffer" in checkpoint:
            agent.replay_buffer.buffer = checkpoint["replay_buffer_buffer"]
            agent.replay_buffer.position = checkpoint["replay_buffer_position"]
            agent.replay_buffer.priorities = checkpoint["replay_buffer_priorities"]
            agent.replay_buffer.max_priority = checkpoint["replay_buffer_max_priority"]
            agent.replay_buffer.to_device(agent.device)

        # Load scheduler state
        if "scheduler_returns_history" in checkpoint:
            from collections import deque
            agent.scheduler.returns_history = deque(
                checkpoint["scheduler_returns_history"],
                maxlen=agent.scheduler.window_size
            )
            agent.scheduler.last_growth_episode = checkpoint.get("scheduler_last_growth_episode", 0)

        # Load encoder if present
        if agent.encoder and "encoder_state_dict" in checkpoint:
            agent.encoder.load_state_dict(checkpoint["encoder_state_dict"])
            agent.encoder_optimizer.load_state_dict(checkpoint["encoder_optimizer_state_dict"])

        print(f"Loaded checkpoint from episode {checkpoint['episode']}")
        print(f"Current resolution: {agent.action_discretizer.current_bins} bins")
        print(f"Growth history: {agent.growth_history}")

        return checkpoint["episode"]

    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return 0


def train_growing_qn():
    """Train Growing Q-Networks agent."""
    args = parse_gqn_args()
    config = GQNConfig.get_walker_config(args)

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    # Environment setup
    if config.task not in [f"{domain}_{task}" for domain, task in suite.ALL_TASKS]:
        print(f"Warning: Task {config.task} not found in suite, using walker_walk")
        config.task = "walker_walk"

    domain_name, task_name = config.task.split("_", 1)
    env = suite.load(domain_name, task_name)
    action_spec = env.action_spec()
    obs_spec = env.observation_spec()

    # Observation shape
    if config.use_pixels:
        obs_shape = (3, 84, 84)
    else:
        state_dim = sum(spec.shape[0] if len(spec.shape) > 0 else 1 for spec in obs_spec.values())
        obs_shape = (state_dim,)

    action_spec_dict = {"low": action_spec.minimum, "high": action_spec.maximum}

    # Create agent
    agent = GrowingQNAgent(config, obs_shape, action_spec_dict)

    # Create output directories
    os.makedirs("output/checkpoints", exist_ok=True)
    os.makedirs("output/metrics", exist_ok=True)
    os.makedirs("output/plots", exist_ok=True)

    # Load checkpoint if specified
    start_episode = 0
    if args.load_checkpoints:
        start_episode = load_checkpoint(agent, args.load_checkpoints)
        start_episode += 1

    # Metrics tracking
    metrics_tracker = MetricsTracker(save_dir="output/metrics")
    if start_episode > 0:
        metrics_tracker.load_metrics()

    print(f"Growing Q-Networks Agent Setup:")
    print(f"  Task: {config.task}")
    print(f"  Decouple: {agent.config.decouple}")
    print(f"  Action dimensions: {agent.action_discretizer.action_dim}")
    print(f"  Growing schedule: {config.growing_schedule}")
    print(f"  Growth sequence: {agent.action_discretizer.growth_sequence}")
    print(f"  Current bins: {agent.action_discretizer.current_bins}")
    print(f"  Action penalty: {config.action_penalty}")
    print(f"Starting training from episode {start_episode}...")

    start_time = time.time()

    for episode in range(start_episode, config.num_episodes):
        episode_start_time = time.time()
        episode_reward = 0
        original_episode_reward = 0  # Track reward without penalty
        episode_loss = 0
        episode_q_mean = 0
        loss_count = 0
        action_magnitude_sum = 0

        time_step = env.reset()
        obs = process_observation(time_step.observation, config.use_pixels, device)
        agent.observe_first(obs)

        step = 0
        while not time_step.last():
            # Select action
            action = agent.select_action(obs)
            action_np = action.cpu().numpy()

            # Take environment step
            time_step = env.step(action_np)
            next_obs = process_observation(time_step.observation, config.use_pixels, device)

            # Get reward and apply penalty
            original_reward = time_step.reward if time_step.reward is not None else 0.0
            reward = apply_action_penalty(original_reward, action_np, config.action_penalty)
            done = time_step.last()

            # Store transition
            agent.observe(action, reward, next_obs, done)

            # Update agent
            if len(agent.replay_buffer) > config.min_replay_size:
                metrics = agent.update()
                if metrics:
                    if "loss" in metrics and metrics["loss"] is not None:
                        episode_loss += metrics["loss"]
                    episode_q_mean += metrics.get("q1_mean", 0)
                    loss_count += 1

            # Track metrics
            obs = next_obs
            episode_reward += reward
            original_episode_reward += original_reward
            action_magnitude_sum += np.linalg.norm(action_np)
            step += 1

            if step > 1000:  # Episode length limit
                break

        # End episode and potentially grow action space
        agent.end_episode(original_episode_reward)  # Use original reward for growth decisions

        # Calculate averages
        avg_loss = episode_loss / max(loss_count, 1) if loss_count > 0 else 0.0
        avg_q_mean = episode_q_mean / max(loss_count, 1)
        avg_action_magnitude = action_magnitude_sum / max(step, 1)

        # Log metrics
        metrics_tracker.log_episode(
            episode=episode,
            reward=episode_reward,
            length=step,
            loss=avg_loss if loss_count > 0 else None,
            q_mean=avg_q_mean if loss_count > 0 else None,
            epsilon=agent.epsilon,
        )

        # Update epsilon
        agent.update_epsilon(decay_rate=0.995, min_epsilon=0.01)

        # Logging
        episode_time = time.time() - episode_start_time

        if episode % args.log_interval == 0:
            growth_info = agent.get_growth_info()
            print(
                f"Episode {episode:4d} | "
                f"Reward: {episode_reward:7.2f} | "
                f"Original: {original_episode_reward:7.2f} | "
                f"Loss: {avg_loss:8.6f} | "
                f"Q-mean: {avg_q_mean:6.3f} | "
                f"Bins: {growth_info['current_bins']} | "
                f"Action_mag: {avg_action_magnitude:.3f} | "
                f"Time: {episode_time:.2f}s"
            )

        # Detailed logging
        if episode % args.detailed_log_interval == 0 and episode > 0:
            elapsed_time = time.time() - start_time
            avg_episode_time = elapsed_time / (episode - start_episode + 1)
            eta = avg_episode_time * (config.num_episodes - episode - 1)

            growth_info = agent.get_growth_info()
            recent_rewards = metrics_tracker.episode_rewards[-args.detailed_log_interval:]

            print(f"\n--- Episode {episode} Detailed Summary ---")
            print(f"Penalized Reward: {episode_reward:.2f}")
            print(f"Original Reward: {original_episode_reward:.2f}")
            print(f"Recent {args.detailed_log_interval} episodes avg reward: {np.mean(recent_rewards):.2f}")
            print(f"Current resolution: {growth_info['current_bins']} bins")
            print(f"Growth history: {growth_info['growth_history']}")
            print(f"Buffer size: {len(agent.replay_buffer)}")
            print(f"Elapsed: {elapsed_time / 60:.1f}min | ETA: {eta / 60:.1f}min")
            print("-" * 40)

        # Memory cleanup
        if episode % 10 == 0 and device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

        # Save checkpoints
        if episode % args.checkpoint_interval == 0:
            metrics_tracker.save_metrics()
            checkpoint_path = f"output/checkpoints/gqn_episode_{episode}.pth"
            save_checkpoint(agent, episode, checkpoint_path, metrics_tracker)
            print(f"Checkpoint saved: {checkpoint_path}")

    # Final save
    metrics_tracker.save_metrics()
    final_checkpoint = "output/checkpoints/gqn_final.pth"
    save_checkpoint(agent, config.num_episodes, final_checkpoint, metrics_tracker)
    print(f"Final checkpoint saved: {final_checkpoint}")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time / 60:.1f} minutes!")

    # Final growth summary
    growth_info = agent.get_growth_info()
    print(f"\nGrowing Q-Networks Summary:")
    print(f"  Final resolution: {growth_info['current_bins']} bins")
    print(f"  Growth sequence achieved: {growth_info['growth_history']}")
    print(f"  Total resolution levels: {len(growth_info['growth_history'])}")

    # Create and save plots
    print("\nGenerating plots...")
    plotter = PlottingUtils(metrics_tracker, save_dir="output/plots")
    plotter.plot_training_curves(save=True)
    plotter.plot_reward_distribution(save=True)
    plotter.print_summary_stats()

    return agent


if __name__ == "__main__":
    agent = train_growing_qn()