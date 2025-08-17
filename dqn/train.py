import numpy as np
import torch
import os
from config import Config
from agents import DecQNAgent



def train_decqn():
    """training function."""
    config = Config()
    config.algorithm = 'decqnvis'  # Use decoupled + vision
    config.use_double_q = True
    config.num_episodes = 1000

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

    agent = DecQNAgent(config, obs_shape, action_spec, device=device)

    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)

    print(f"Agent setup - decouple: {agent.config.decouple}, action_dim: {agent.action_discretizer.action_dim}")

    # Training loop (mock - replace with actual environment interaction)
    print("Starting training...")
    for episode in range(config.num_episodes):
        # Mock episode - replace with actual environment interaction
        episode_reward = 0

        # Generate mock observations as tensors directly on device
        if config.use_pixels:
            # Mock RGB image data [0, 255]
            obs = torch.randint(0, 256, obs_shape, device=device, dtype=torch.float32)
        else:
            # Mock state data
            obs = torch.randn(obs_shape, device=device, dtype=torch.float32)

        agent.observe_first(obs)

        for step in range(100):  # Mock episode length
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

            obs = next_obs
            episode_reward += reward

        # Log progress
        if episode % 5 == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.2f}")

        # Clear GPU cache periodically
        if episode % 10 == 0 and device == 'cuda':
            torch.cuda.empty_cache()

        # Save checkpoint
        if episode % 100 == 0:
            checkpoint_path = f'./checkpoints/decqn_episode_{episode}.pth'
            save_checkpoint(agent, episode, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    # Save final checkpoint
    final_checkpoint = 'checkpoints/decqn_final.pth'
    save_checkpoint(agent, config.num_episodes, final_checkpoint)
    print(f"Final checkpoint saved: {final_checkpoint}")

    print("Training completed!")
    return agent

def save_checkpoint(agent, episode, path):
    """Save agent checkpoint."""
    checkpoint = {
        'episode': episode,
        'q_network_state_dict': agent.q_network.state_dict(),
        'target_q_network_state_dict': agent.target_q_network.state_dict(),
        'q_optimizer_state_dict': agent.q_optimizer.state_dict(),
        'config': agent.config,
        'training_step': agent.training_step
    }

    if agent.encoder:
        checkpoint['encoder_state_dict'] = agent.encoder.state_dict()
        checkpoint['encoder_optimizer_state_dict'] = agent.encoder_optimizer.state_dict()

    torch.save(checkpoint, path)


if __name__ == "__main__":
    agent = train_decqn()