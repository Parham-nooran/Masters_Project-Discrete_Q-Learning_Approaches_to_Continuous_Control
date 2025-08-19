import numpy as np
import torch
import argparse
import os
from agents import DecQNAgent


def load_checkpoint(checkpoint_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Load agent from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    # Mock environment setup - should match training setup
    obs_shape = (3, 84, 84) if config.use_pixels else (17,)
    action_spec = {'low': np.array([-1.0, -1.0]), 'high': np.array([1.0, 1.0])}

    # Create agent
    agent = DecQNAgent(config, obs_shape, action_spec, device=device)

    # Load state dictionaries
    agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
    agent.target_q_network.load_state_dict(checkpoint['target_q_network_state_dict'])
    agent.q_optimizer.load_state_dict(checkpoint['q_optimizer_state_dict'])
    agent.training_step = checkpoint.get('training_step', 0)

    if agent.encoder and 'encoder_state_dict' in checkpoint:
        agent.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        agent.encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])

    print(f"Loaded checkpoint from episode {checkpoint['episode']}")
    return agent


def demonstrate(checkpoint_path, num_episodes=5, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Run demonstration episodes."""
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Using device: {device}")

    agent = load_checkpoint(checkpoint_path, device)

    # Mock environment setup - replace with actual environment
    obs_shape = (3, 84, 84) if agent.config.use_pixels else (17,)

    print(f"\nRunning {num_episodes} demonstration episodes...")

    total_rewards = []

    for episode in range(num_episodes):
        episode_reward = 0

        # Mock episode - replace with actual environment interaction
        obs = np.random.random(obs_shape).astype(np.float32)

        print(f"\nEpisode {episode + 1}:")

        for step in range(100):  # Mock episode length
            # Select action using evaluation policy (no exploration)
            action = agent.select_action(obs, evaluate=True)

            # Mock environment step - replace with actual environment
            next_obs = np.random.random(obs_shape).astype(np.float32)
            reward = np.random.random() - 0.5  # Mock reward
            done = step == 99

            obs = next_obs
            episode_reward += reward

            # Print action every 20 steps
            if step % 20 == 0:
                print(f"  Step {step}: Action = {action}, Reward = {reward:.3f}")

            if done:
                break

        total_rewards.append(episode_reward)
        print(f"  Episode {episode + 1} total reward: {episode_reward:.3f}")

    # Summary statistics
    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)

    print(f"\nDemonstration Summary:")
    print(f"Episodes: {num_episodes}")
    print(f"Mean reward: {mean_reward:.3f}")
    print(f"Std reward: {std_reward:.3f}")
    print(f"Min reward: {np.min(total_rewards):.3f}")
    print(f"Max reward: {np.max(total_rewards):.3f}")


def main():
    parser = argparse.ArgumentParser(description='Demonstrate trained DecQN agent')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/decqn_final.pth',
                        help='Path to checkpoint file')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of demonstration episodes')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cpu, cuda, or auto)')

    args = parser.parse_args()

    # Device selection
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    demonstrate(args.checkpoint, args.episodes, device)


if __name__ == "__main__":
    main()