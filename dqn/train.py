import time
import gc
import torch
from config import *

from agents import DecQNAgent
from plotting_utils import *


def train_decqn():
    """training function."""
    args = parse_args()
    config = create_config_from_args(args)

    # Device selection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # GPU optimizations
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    # Mock environment setup - replace with actual environment
    obs_shape = (3, 84, 84) if config.use_pixels else (17,)  # Walker state dim
    action_spec = {'low': np.array([-1.0, -1.0]), 'high': np.array([1.0, 1.0])}  # 2D continuous

    agent = DecQNAgent(config, obs_shape, action_spec)

    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)

    # Handle checkpoint loading
    start_episode = 0
    checkpoint_to_load = args.load_checkpoint

    # If no specific checkpoint provided, check for latest
    if checkpoint_to_load is None:
        latest_checkpoint = find_latest_checkpoint()
        if latest_checkpoint:
            checkpoint_to_load = latest_checkpoint
            print(f"Found latest checkpoint: {latest_checkpoint}")

    # Load checkpoint if available
    if checkpoint_to_load:
        if os.path.exists(checkpoint_to_load):
            try:
                loaded_episode = agent.load_checkpoint(checkpoint_to_load)
                start_episode = loaded_episode + 1
                print(f"Resumed from episode {loaded_episode}, starting at episode {start_episode}")
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")
                print("Starting fresh training...")
                start_episode = 0
        else:
            print(f"Checkpoint file {checkpoint_to_load} not found. Starting fresh training...")

    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(save_dir="metrics")

    # Load existing metrics if resuming
    if start_episode > 0:
        metrics_tracker.load_metrics()

    print(f"Agent setup - decouple: {agent.config.decouple}, action_dim: {agent.action_discretizer.action_dim}")

    # [Rest of the training loop remains the same...]
    # Training loop (mock - replace with actual environment interaction)
    print("Starting training...")
    start_time = time.time()
    for episode in range(start_episode, config.num_episodes):
        episode_start_time = time.time()
        # Mock episode - replace with actual environment interaction
        episode_reward = 0
        episode_loss = 0
        episode_q_mean = 0
        loss_count = 0

        # Generate mock observations as tensors directly on device
        if config.use_pixels:
            # Mock RGB image data [0, 255]
            obs = torch.randint(0, 256, obs_shape, device=device, dtype=torch.float32)
        else:
            # Mock state data
            obs = torch.randn(obs_shape, device=device, dtype=torch.float32)

        agent.observe_first(obs)

        for step in range(config.mock_episode_length):  # Mock episode length
            action = agent.select_action(obs)

            # Generate next observation
            if config.use_pixels:
                next_obs = torch.randint(0, 256, obs_shape, device=device, dtype=torch.float32)
            else:
                next_obs = torch.randn(obs_shape, device=device, dtype=torch.float32)

            reward = torch.rand(1, device=device).item() - 0.5  # Mock reward
            done = step == 99

            agent.observe(action, reward, next_obs, done)

            # Update agent
            if len(agent.replay_buffer) > config.min_replay_size:
                metrics = agent.update()
                if metrics:
                    episode_loss += metrics.get('loss', 0)
                    episode_q_mean += metrics.get('q1_mean', 0)
                    loss_count += 1

            obs = next_obs
            episode_reward += reward

        # Calculate averages
        avg_loss = episode_loss / max(loss_count, 1)
        avg_q_mean = episode_q_mean / max(loss_count, 1)

        # Log metrics
        metrics_tracker.log_episode(
            episode=episode,
            reward=episode_reward,
            length=config.mock_episode_length,
            loss=avg_loss if loss_count > 0 else None,
            q_mean=avg_q_mean if loss_count > 0 else None,
            epsilon=agent.epsilon
        )

        agent.update_epsilon(decay_rate=0.995, min_epsilon=0.01)

        # Enhanced progress logging
        episode_time = time.time() - episode_start_time
        if episode % args.log_interval == 0:
            print(f"Episode {episode:4d} | "
                  f"Reward: {episode_reward:7.2f} | "
                  f"Loss: {avg_loss:8.6f} | "
                  f"Q-mean: {avg_q_mean:6.3f} | "
                  f"Time: {episode_time:.2f}s | "
                  f"Buffer: {len(agent.replay_buffer):6d}")
            torch.cuda.empty_cache()
            gc.collect()

        # Detailed logging every N episodes
        if episode % args.detailed_log_interval == 0 and episode > 0:
            elapsed_time = time.time() - start_time
            avg_episode_time = elapsed_time / (episode - start_episode + 1)
            eta = avg_episode_time * (config.num_episodes - episode - 1)
            print(f"\n--- Episode {episode} Summary ---")
            print(f"Cumulative Reward: {episode_reward:.2f}")
            print(
                f"Recent {args.detailed_log_interval} episodes avg reward: {np.mean(metrics_tracker.episode_rewards[-args.detailed_log_interval:]):.2f}")
            print(f"Elapsed Time: {elapsed_time / 60:.1f} min | ETA: {eta / 60:.1f} min")
            print("-" * 35)

        # Clear GPU cache periodically
        if episode % 10 == 0 and device == 'cuda':
            torch.cuda.empty_cache()

        # Save checkpoint
        if episode % args.checkpoint_interval == 0:
            import shutil
            # Save metrics before clearing checkpoints
            metrics_tracker.save_metrics()

            # Remove old checkpoints directory and create new one
            if os.path.exists('checkpoints'):
                shutil.rmtree('checkpoints')
            os.makedirs('checkpoints', exist_ok=True)

            checkpoint_path = f'./checkpoints/decqn_episode_{episode}.pth'
            save_checkpoint(agent, episode, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    # Save final checkpoint
    import shutil
    metrics_tracker.save_metrics()  # Save metrics first
    if os.path.exists('checkpoints'):
        shutil.rmtree('checkpoints')
    os.makedirs('checkpoints', exist_ok=True)
    final_checkpoint = 'checkpoints/decqn_final.pth'
    save_checkpoint(agent, config.num_episodes, final_checkpoint)
    print(f"Final checkpoint saved: {final_checkpoint}")

    metrics_tracker.save_metrics()
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time / 60:.1f} minutes!")

    print("Training completed!")

    # Create and save plots
    print("Plotting ...")
    plotter = PlottingUtils(metrics_tracker)
    plotter.plot_training_curves(save=True)
    plotter.plot_reward_distribution(save=True)
    plotter.print_summary_stats()

    return agent


def find_latest_checkpoint(checkpoint_dir='checkpoints'):
    """Find the latest checkpoint file."""
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not checkpoint_files:
        return None

    # Sort by modification time and get the latest
    checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
    latest_file = checkpoint_files[0]

    return os.path.join(checkpoint_dir, latest_file)


def save_checkpoint(agent, episode, path):
    """Save agent checkpoint."""
    # Convert config to dict to avoid pickle issues
    config_dict = vars(agent.config)  # Much simpler since it's a SimpleNamespace

    checkpoint = {
        'episode': episode,
        'q_network_state_dict': agent.q_network.state_dict(),
        'target_q_network_state_dict': agent.target_q_network.state_dict(),
        'q_optimizer_state_dict': agent.q_optimizer.state_dict(),
        'config': agent.config,
        'training_step': agent.training_step,
        'epsilon': agent.epsilon,
        'replay_buffer_buffer': agent.replay_buffer.buffer,
        'replay_buffer_position': agent.replay_buffer.position,
        'replay_buffer_priorities': agent.replay_buffer.priorities,
        'replay_buffer_max_priority': agent.replay_buffer.max_priority
    }

    if agent.encoder:
        checkpoint['encoder_state_dict'] = agent.encoder.state_dict()
        checkpoint['encoder_optimizer_state_dict'] = agent.encoder_optimizer.state_dict()

    torch.save(checkpoint, path)


if __name__ == "__main__":
    agent = train_decqn()