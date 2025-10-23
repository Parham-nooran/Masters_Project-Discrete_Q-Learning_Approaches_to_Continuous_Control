import argparse
import os
from datetime import datetime

import cv2
import numpy as np
import torch
from dm_control import suite

from src.deqn.agent import DecQNAgent
from src.deqn.train import process_observation


def load_checkpoint(
    checkpoint_path, env, device="cuda" if torch.cuda.is_available() else "cpu"
):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config.py"]

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

    agent = DecQNAgent(config, obs_shape, action_spec_dict)

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
    save_video=False,
    video_path=None,
    show_display=True,
    fps=30,
):
    print(f"Loading checkpoints: {checkpoint_path}")
    print(f"Using device: {device}")

    env = suite.load(domain_name="walker", task_name="walk")
    agent = load_checkpoint(checkpoint_path, env, device)

    print(f"\nRunning {num_episodes} demonstration episodes...")

    video_writer = None
    if save_video:
        if video_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = f"./output/videos/decqn_demo_{timestamp}.mp4"

        os.makedirs(
            os.path.dirname(video_path) if os.path.dirname(video_path) else ".",
            exist_ok=True,
        )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (640, 480))
        print(f"Recording video to: {video_path}")

    total_rewards = []

    for episode in range(num_episodes):
        episode_reward = 0

        time_step = env.reset()
        obs = process_observation(
            time_step.observation, agent.config.use_pixels, device
        )

        print(f"\nEpisode {episode + 1}:")

        step = 0
        action_np = None
        while not time_step.last() and step < 1000:
            try:
                pixels = env.physics.render(width=640, height=480, camera_id=0)
                if pixels is None or len(pixels.shape) != 3:
                    pixels = env.physics.render(width=640, height=480, camera_id=0)
            except Exception as e:
                print(f"Caught exception {e}")
                pixels = np.zeros((480, 640, 3), dtype=np.uint8)

            frame = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)

            info_text = [
                f"Episode: {episode + 1}/{num_episodes}",
                f"Step: {step}",
                f"Reward: {episode_reward:.2f}",
                f"Last Action: {action_np if action_np is not None else 'None'}",
            ]

            for i, text in enumerate(info_text):
                cv2.putText(
                    frame,
                    text,
                    (10, 30 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

            if save_video and video_writer is not None:
                video_writer.write(frame)

            if show_display:
                cv2.imshow("DecQN Walker Demo", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("Demo terminated by user")
                    break

            action = agent.select_action(obs, evaluate=True)
            if isinstance(action, torch.Tensor):
                action_np = action.cpu().numpy()
            else:
                action_np = action

            time_step = env.step(action_np)
            obs = process_observation(
                time_step.observation, agent.config.use_pixels, device
            )

            reward = time_step.reward if time_step.reward is not None else 0.0
            episode_reward += reward
            step += 1
        total_rewards.append(episode_reward)
        print(
            f"  Episode {episode + 1} total reward: {episode_reward:.3f} ({step} steps)"
        )

    if save_video and video_writer is not None:
        video_writer.release()
        print(f"Video saved to: {video_path}")

    if show_display:
        cv2.destroyAllWindows()

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
        default="checkpoints/decqn_episode_500.pth",
        help="Path to checkpoints file",
    )
    parser.add_argument(
        "--episodes", type=int, default=5, help="Number of demonstration episodes"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to use (cpu, cuda, or auto)"
    )
    parser.add_argument(
        "--save-video", default=True, action="store_true", help="Save video to file"
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default=None,
        help="Path to save video (default: auto-generated)",
    )
    parser.add_argument(
        "--no-display",
        default=False,
        action="store_true",
        help="Don't show video display window",
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Video frame rate (default: 30)"
    )

    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    demonstrate(
        args.checkpoints,
        args.episodes,
        device,
        save_video=args.save_video,
        video_path=args.video_path,
        show_display=not args.no_display,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
