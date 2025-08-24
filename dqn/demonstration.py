import numpy as np
import torch
import argparse
import os
from agents import DecQNAgent
from dm_control import suite


def process_observation(dm_obs, use_pixels, device):
    """Convert DM Control observation to tensor format."""
    if use_pixels:
        # Get RGB camera observation
        if "pixels" in dm_obs:
            obs = dm_obs["pixels"]
        else:
            # Some DM Control tasks use different camera names
            camera_obs = [v for k, v in dm_obs.items() if "camera" in k or "rgb" in k]
            if camera_obs:
                obs = camera_obs[0]
            else:
                raise ValueError(
                    "No pixel observations found in DM Control observation"
                )

        # Convert to CHW format and normalize
        obs = torch.tensor(obs, dtype=torch.float32, device=device)
        if len(obs.shape) == 3:  # HWC -> CHW
            obs = obs.permute(2, 0, 1)

        return obs
    else:
        # Concatenate all state observations
        state_parts = []
        for key in sorted(dm_obs.keys()):  # Consistent ordering
            val = dm_obs[key]
            if isinstance(val, np.ndarray):
                state_parts.append(val.flatten())
            else:
                state_parts.append(np.array([val], dtype=np.float32))

        state_vector = np.concatenate(state_parts)
        return torch.tensor(state_vector, dtype=torch.float32, device=device)


def load_checkpoint(
    checkpoint_path, env, device="cuda" if torch.cuda.is_available() else "cpu"
):
    """Load agent from checkpoints."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    # Setup DM Control environment to get correct specs
    action_spec = env.action_spec()
    obs_spec = env.observation_spec()

    # Convert DM Control specs to your format
    if config.use_pixels:
        obs_shape = (3, 84, 84)  # RGB camera view
    else:
        # Calculate state dimension from observation spec
        state_dim = sum(
            spec.shape[0] if len(spec.shape) > 0 else 1 for spec in obs_spec.values()
        )
        obs_shape = (state_dim,)

    # Convert action spec to dict format
    action_spec_dict = {"low": action_spec.minimum, "high": action_spec.maximum}

    # Create agent
    agent = DecQNAgent(config, obs_shape, action_spec_dict)

    # Load state dictionaries
    agent.q_network.load_state_dict(checkpoint["q_network_state_dict"])
    agent.target_q_network.load_state_dict(checkpoint["target_q_network_state_dict"])
    agent.q_optimizer.load_state_dict(checkpoint["q_optimizer_state_dict"])
    agent.training_step = checkpoint.get("training_step", 0)

    if agent.encoder and "encoder_state_dict" in checkpoint:
        agent.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        agent.encoder_optimizer.load_state_dict(
            checkpoint["encoder_optimizer_state_dict"]
        )

    print(f"Loaded checkpoints from episode {checkpoint['episode']}")
    return agent


def demonstrate(
    checkpoint_path,
    num_episodes=5,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """Run demonstration episodes."""
    print(f"Loading checkpoints: {checkpoint_path}")
    print(f"Using device: {device}")

    env = suite.load(domain_name="walker", task_name="walk")
    agent = load_checkpoint(checkpoint_path, env, device)

    print(f"\nRunning {num_episodes} demonstration episodes...")

    total_rewards = []

    for episode in range(num_episodes):
        episode_reward = 0

        # Reset environment
        time_step = env.reset()
        obs = process_observation(
            time_step.observation, agent.config.use_pixels, device
        )

        print(f"\nEpisode {episode + 1}:")

        step = 0
        while not time_step.last() and step < 1000:
            # Select action using evaluation policy
            action = agent.select_action(obs, evaluate=True)

            # Convert to numpy for DM Control
            if isinstance(action, torch.Tensor):
                action_np = action.cpu().numpy()
            else:
                action_np = action

            # Step environment
            time_step = env.step(action_np)
            obs = process_observation(
                time_step.observation, agent.config.use_pixels, device
            )

            reward = time_step.reward if time_step.reward is not None else 0.0
            episode_reward += reward
            step += 1

            # Print action every 50 steps
            if step % 50 == 0:
                print(f"  Step {step}: Action = {action_np}, Reward = {reward:.3f}")

        total_rewards.append(episode_reward)
        print(
            f"  Episode {episode + 1} total reward: {episode_reward:.3f} ({step} steps)"
        )

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
    parser = argparse.ArgumentParser(description="Demonstrate trained DecQN agent")
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="checkpoints/decqn_final.pth",
        help="Path to checkpoints file",
    )
    parser.add_argument(
        "--episodes", type=int, default=5, help="Number of demonstration episodes"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (cpu, cuda, or auto)"
    )

    args = parser.parse_args()

    # Device selection
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    demonstrate(args.checkpoint, args.episodes, device)


if __name__ == "__main__":
    main()
