import torch
import argparse
import os
from datetime import datetime
import cv2
from dm_control import suite
import numpy as np
from src.gqn.agent import GrowingQNAgent
from src.common.utils import process_observation


def load_gqn_checkpoint(
    checkpoint_path, env, device="cuda" if torch.cuda.is_available() else "cpu"
):
    """Load GQN checkpoint with proper state restoration."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    action_spec = env.action_spec()
    obs_spec = env.observation_spec()

    if config.use_pixels:
        obs_shape = (3, 84, 84)
    else:
        state_dim = sum(
            spec.shape[0] if len(spec.shape) > 0 else 1 for spec in obs_spec.values()
        )
        obs_shape = (state_dim,)

    action_spec_dict = {"low": action_spec.minimum, "high": action_spec.maximum}
    agent = GrowingQNAgent(config, obs_shape, action_spec_dict)
    agent.q_network.load_state_dict(checkpoint["q_network_state_dict"])
    agent.target_q_network.load_state_dict(checkpoint["target_q_network_state_dict"])
    agent.q_optimizer.load_state_dict(checkpoint["q_optimizer_state_dict"])
    agent.training_step = checkpoint.get("training_step", 0)
    agent.epsilon = checkpoint.get("epsilon", 0.0)
    agent.episode_count = checkpoint.get("episode_count", 0)

    if "current_resolution_level" in checkpoint:
        agent.current_resolution_level = checkpoint["current_resolution_level"]
    if "growth_history" in checkpoint:
        agent.growth_history = checkpoint["growth_history"]
    if "action_discretizer_current_bins" in checkpoint:
        agent.action_discretizer.current_bins = checkpoint[
            "action_discretizer_current_bins"
        ]
        agent.action_discretizer.action_bins = agent.action_discretizer.all_action_bins[
            agent.action_discretizer.current_bins
        ]
    if "action_discretizer_current_growth_idx" in checkpoint:
        agent.action_discretizer.current_growth_idx = checkpoint[
            "action_discretizer_current_growth_idx"
        ]

    if agent.encoder and "encoder_state_dict" in checkpoint:
        agent.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        agent.encoder_optimizer.load_state_dict(
            checkpoint["encoder_optimizer_state_dict"]
        )

    print(f"Loaded GQN checkpoint from episode {checkpoint['episode']}")
    print(f"Final resolution: {agent.action_discretizer.current_bins} bins")
    print(f"Growth history: {agent.growth_history}")

    return agent


def demonstrate_gqn(
    checkpoint_path,
    num_episodes=5,
    device="cuda" if torch.cuda.is_available() else "cpu",
    save_video=False,
    video_path=None,
    show_display=True,
    fps=30,
    task="walker_walk",
):
    """Demonstrate trained Growing Q-Networks agent."""
    print(f"Loading GQN checkpoint: {checkpoint_path}")
    print(f"Using device: {device}")
    print(f"Task: {task}")

    domain_name, task_name = task.split("_", 1)
    env = suite.load(domain_name, task_name)

    agent = load_gqn_checkpoint(checkpoint_path, env, device)

    print(f"\nRunning {num_episodes} demonstration episodes...")

    # Video recording setup
    video_writer = None
    if save_video:
        if video_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = f"./output/videos/gqn_demo_{timestamp}.mp4"

        os.makedirs(
            os.path.dirname(video_path) if os.path.dirname(video_path) else ".",
            exist_ok=True,
        )
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (640, 480))
        print(f"Recording video to: {video_path}")

    total_rewards = []
    action_magnitudes = []
    action_changes = []

    for episode in range(num_episodes):
        episode_reward = 0
        episode_action_magnitude = 0
        episode_action_changes = []

        time_step = env.reset()
        obs = process_observation(
            time_step.observation, agent.config.use_pixels, device
        )
        agent.observe_first(obs)

        print(f"\nEpisode {episode + 1}:")

        step = 0
        last_action = None

        while not time_step.last() and step < 1000:
            try:
                # Render environment
                pixels = env.physics.render(width=640, height=480, camera_id=0)
                if pixels is None or len(pixels.shape) != 3:
                    pixels = env.physics.render(width=640, height=480, camera_id=0)
            except Exception as e:
                print(f"Rendering error: {e}")
                pixels = np.zeros((480, 640, 3), dtype=np.uint8)

            frame = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)

            # Get action from agent (no exploration)
            action = agent.select_action(obs, evaluate=True)
            action_np = (
                action.cpu().numpy() if isinstance(action, torch.Tensor) else action
            )

            # Calculate action metrics
            action_mag = np.linalg.norm(action_np)
            episode_action_magnitude += action_mag

            if last_action is not None:
                action_change = np.linalg.norm(action_np - last_action)
                episode_action_changes.append(action_change)

            last_action = action_np.copy()

            # Add information overlay
            growth_info = agent.get_growth_info()
            info_text = [
                f"Episode: {episode + 1}/{num_episodes}",
                f"Step: {step}",
                f"Reward: {episode_reward:.2f}",
                f"Current Bins: {growth_info['current_bins']}",
                f"Growth History: {growth_info['growth_history']}",
                f"Action: [{', '.join([f'{x:.3f}' for x in action_np])}]",
                f"Action Mag: {action_mag:.3f}",
            ]

            for i, text in enumerate(info_text):
                # Use smaller font for more info
                font_scale = 0.5 if i >= 4 else 0.6
                cv2.putText(
                    frame,
                    text,
                    (10, 25 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    2,
                )

            # Save frame to video if recording
            if save_video and video_writer is not None:
                video_writer.write(frame)

            # Display frame if requested
            if show_display:
                cv2.imshow("Growing Q-Networks Demo", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("Demo terminated by user")
                    break

            # Take environment step
            time_step = env.step(action_np)
            obs = process_observation(
                time_step.observation, agent.config.use_pixels, device
            )

            reward = time_step.reward if time_step.reward is not None else 0.0
            episode_reward += reward
            step += 1

        # Episode summary
        avg_action_mag = episode_action_magnitude / max(step, 1)
        avg_action_change = (
            np.mean(episode_action_changes) if episode_action_changes else 0.0
        )

        total_rewards.append(episode_reward)
        action_magnitudes.append(avg_action_mag)
        action_changes.append(avg_action_change)

        print(
            f"  Episode {episode + 1} total reward: {episode_reward:.3f} ({step} steps)"
        )
        print(f"  Average action magnitude: {avg_action_mag:.3f}")
        print(f"  Average action change: {avg_action_change:.3f}")

    # Cleanup
    if save_video and video_writer is not None:
        video_writer.release()
        print(f"Video saved to: {video_path}")

    if show_display:
        cv2.destroyAllWindows()

    # Final statistics
    print(f"\n=== Growing Q-Networks Demonstration Summary ===")
    print(f"Episodes: {num_episodes}")
    print(f"Mean reward: {np.mean(total_rewards):.3f}")
    print(f"Std reward: {np.std(total_rewards):.3f}")
    print(f"Min reward: {np.min(total_rewards):.3f}")
    print(f"Max reward: {np.max(total_rewards):.3f}")
    print(f"Mean action magnitude: {np.mean(action_magnitudes):.3f}")
    print(f"Mean action change: {np.mean(action_changes):.3f}")

    # Growth information
    growth_info = agent.get_growth_info()
    print(f"\n=== Action Space Growth Summary ===")
    print(f"Final resolution: {growth_info['current_bins']} bins per dimension")
    print(f"Growth sequence: {growth_info['growth_sequence']}")
    print(f"Achieved growth: {growth_info['growth_history']}")
    print(f"Maximum possible resolution: {growth_info['max_bins']} bins")

    return total_rewards, action_magnitudes, action_changes


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(
        description="Demonstrate trained Growing Q-Networks agent"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="output/checkpoints/gqn_final.pth",
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--episodes", type=int, default=5, help="Number of demonstration episodes"
    )
    parser.add_argument(
        "--task", type=str, default="walker_walk", help="Environment task"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (cpu, cuda, or auto)"
    )
    parser.add_argument("--save-video", action="store_true", help="Save video to file")
    parser.add_argument(
        "--video-path",
        type=str,
        default=None,
        help="Path to save video (default: auto-generated)",
    )
    parser.add_argument(
        "--no-display", action="store_true", help="Don't show video display window"
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Video frame rate (default: 30)"
    )

    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("Growing Q-Networks Demonstration")
    print("=" * 40)

    try:
        demonstrate_gqn(
            args.checkpoint,
            args.episodes,
            device,
            save_video=args.save_video,
            video_path=args.video_path,
            show_display=not args.no_display,
            fps=args.fps,
            task=args.task,
        )
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
