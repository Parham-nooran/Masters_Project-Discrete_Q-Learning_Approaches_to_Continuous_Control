"""Demonstration script for trained Growing Q-Networks agent."""

import torch
import argparse
import os
from datetime import datetime
import cv2
import numpy as np
from dm_control import suite

from src.gqn.agent import GrowingQNAgent
from src.common.observation_utils import process_observation


def load_gqn_checkpoint(
    checkpoint_path, env, device="cuda" if torch.cuda.is_available() else "cpu"
):
    """Load GQN checkpoint with proper state restoration."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    obs_shape, action_spec_dict = _get_env_specs(env, config)
    agent = GrowingQNAgent(config, obs_shape, action_spec_dict, checkpoint_path)

    _load_network_states(agent, checkpoint)
    _load_growth_info(agent, checkpoint)
    _load_encoder_if_exists(agent, checkpoint)
    _print_load_info(checkpoint, agent)

    return agent


def _get_env_specs(env, config):
    """Get environment specifications."""
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
    return obs_shape, action_spec_dict


def _load_network_states(agent, checkpoint):
    """Load network state dictionaries."""
    agent.q_network.load_state_dict(checkpoint["q_network_state_dict"])
    agent.target_q_network.load_state_dict(checkpoint["target_q_network_state_dict"])
    agent.q_optimizer.load_state_dict(checkpoint["q_optimizer_state_dict"])
    agent.training_step = checkpoint.get("training_step", 0)
    agent.episode_count = checkpoint.get("episode_count", 0)
    agent.steps_since_growth = checkpoint.get("steps_since_growth", 0)


def _load_growth_info(agent, checkpoint):
    """Load growth-related information."""
    if "growth_history" in checkpoint:
        agent.growth_history = checkpoint["growth_history"]

    if "action_discretizer_current_bins" in checkpoint:
        agent.action_discretizer.num_bins = checkpoint[
            "action_discretizer_current_bins"
        ]
        agent.action_discretizer.action_bins = agent.action_discretizer.all_action_bins[
            agent.action_discretizer.num_bins
        ]

    if "action_discretizer_current_growth_idx" in checkpoint:
        agent.action_discretizer.current_growth_idx = checkpoint[
            "action_discretizer_current_growth_idx"
        ]


def _load_encoder_if_exists(agent, checkpoint):
    """Load encoder state if it exists."""
    if agent.encoder and "encoder_state_dict" in checkpoint:
        agent.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        agent.encoder_optimizer.load_state_dict(
            checkpoint["encoder_optimizer_state_dict"]
        )


def _print_load_info(checkpoint, agent):
    """Print checkpoint loading information."""
    print(f"Loaded GQN checkpoint from episode {checkpoint['episode']}")
    print(f"Final resolution: {agent.action_discretizer.num_bins} bins")
    print(f"Growth history: {agent.growth_history}")


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
    _print_demo_header(checkpoint_path, device, task)

    env = _load_environment(task)
    agent = load_gqn_checkpoint(checkpoint_path, env, device)

    video_writer = _setup_video_writer(save_video, video_path, fps)

    stats = _run_demonstration_episodes(
        env, agent, num_episodes, video_writer, show_display, device
    )

    _cleanup_video(save_video, video_writer, video_path)
    _print_summary(stats, agent, num_episodes)

    return stats["rewards"], stats["action_magnitudes"], stats["action_changes"]


def _print_demo_header(checkpoint_path, device, task):
    """Print demonstration header."""
    print(f"Loading GQN checkpoint: {checkpoint_path}")
    print(f"Using device: {device}")
    print(f"Task: {task}")
    print(f"\nRunning demonstration episodes...")


def _load_environment(task):
    """Load DM Control environment."""
    domain_name, task_name = task.split("_", 1)
    return suite.load(domain_name, task_name)


def _setup_video_writer(save_video, video_path, fps):
    """Setup video writer if saving video."""
    if not save_video:
        return None

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

    return video_writer


def _run_demonstration_episodes(
    env, agent, num_episodes, video_writer, show_display, device
):
    """Run demonstration episodes and collect statistics."""
    stats = {"rewards": [], "action_magnitudes": [], "action_changes": []}

    for episode in range(num_episodes):
        episode_stats = _run_single_episode(
            env, agent, episode, num_episodes, video_writer, show_display, device
        )

        stats["rewards"].append(episode_stats["reward"])
        stats["action_magnitudes"].append(episode_stats["action_magnitude"])
        stats["action_changes"].append(episode_stats["action_change"])

        _print_episode_summary(episode, num_episodes, episode_stats)

    return stats


def _run_single_episode(
    env, agent, episode, num_episodes, video_writer, show_display, device
):
    """Run a single demonstration episode."""
    time_step = env.reset()
    obs = process_observation(time_step.observation, agent.config.use_pixels, device)
    agent.observe_first(obs)

    episode_stats = _init_episode_stats()
    step = 0
    last_action = None

    while not time_step.last() and step < 1000:
        frame = _render_environment(env)
        action = agent.select_action(obs, evaluate=True)
        action_np = _to_numpy(action)

        _update_action_stats(episode_stats, action_np, last_action)
        last_action = action_np.copy()

        _draw_info_on_frame(
            frame, episode, num_episodes, step, episode_stats, action_np, agent
        )
        _handle_video_and_display(frame, video_writer, show_display)

        time_step = env.step(action_np)
        obs = process_observation(
            time_step.observation, agent.config.use_pixels, device
        )

        reward = time_step.reward if time_step.reward is not None else 0.0
        episode_stats["reward"] += reward
        step += 1

    _finalize_episode_stats(episode_stats, step)
    return episode_stats


def _init_episode_stats():
    """Initialize episode statistics dictionary."""
    return {"reward": 0, "action_magnitude_sum": 0, "action_changes": []}


def _render_environment(env):
    """Render environment frame."""
    try:
        pixels = env.physics.render(width=640, height=480, camera_id=0)
        if pixels is None or len(pixels.shape) != 3:
            pixels = env.physics.render(width=640, height=480, camera_id=0)
    except Exception as e:
        print(f"Rendering error: {e}")
        pixels = np.zeros((480, 640, 3), dtype=np.uint8)

    return cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)


def _to_numpy(action):
    """Convert action to numpy array."""
    if isinstance(action, torch.Tensor):
        return action.cpu().numpy()
    return action


def _update_action_stats(episode_stats, action_np, last_action):
    """Update action statistics."""
    action_mag = np.linalg.norm(action_np)
    episode_stats["action_magnitude_sum"] += action_mag

    if last_action is not None:
        action_change = np.linalg.norm(action_np - last_action)
        episode_stats["action_changes"].append(action_change)


def _draw_info_on_frame(
    frame, episode, num_episodes, step, episode_stats, action_np, agent
):
    """Draw information overlay on frame."""
    growth_info = agent.get_growth_info()
    info_text = [
        f"Episode: {episode + 1}/{num_episodes}",
        f"Step: {step}",
        f"Reward: {episode_stats['reward']:.2f}",
        f"Current Bins: {growth_info['current_bins']}",
        f"Growth History: {growth_info['growth_history']}",
        f"Temperature: {growth_info['temperature']:.4f}",
        f"Action: [{', '.join([f'{x:.3f}' for x in action_np])}]",
    ]

    for i, text in enumerate(info_text):
        font_scale = 0.5 if i >= 5 else 0.6
        cv2.putText(
            frame,
            text,
            (10, 25 + i * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            2,
        )


def _handle_video_and_display(frame, video_writer, show_display):
    """Handle video writing and display."""
    if video_writer is not None:
        video_writer.write(frame)

    if show_display:
        cv2.imshow("Growing Q-Networks Demo", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Demo terminated by user")
            return True

    return False


def _finalize_episode_stats(episode_stats, step):
    """Calculate final episode statistics."""
    episode_stats["action_magnitude"] = episode_stats["action_magnitude_sum"] / max(
        step, 1
    )
    episode_stats["action_change"] = (
        np.mean(episode_stats["action_changes"])
        if episode_stats["action_changes"]
        else 0.0
    )


def _print_episode_summary(episode, num_episodes, episode_stats):
    """Print episode summary statistics."""
    print(f"  Episode {episode + 1} total reward: {episode_stats['reward']:.3f}")
    print(f"  Average action magnitude: {episode_stats['action_magnitude']:.3f}")
    print(f"  Average action change: {episode_stats['action_change']:.3f}")


def _cleanup_video(save_video, video_writer, video_path):
    """Cleanup video writer and close display."""
    if save_video and video_writer is not None:
        video_writer.release()
        print(f"Video saved to: {video_path}")

    cv2.destroyAllWindows()


def _print_summary(stats, agent, num_episodes):
    """Print demonstration summary."""
    print(f"\n=== Growing Q-Networks Demonstration Summary ===")
    print(f"Episodes: {num_episodes}")
    print(f"Mean reward: {np.mean(stats['rewards']):.3f}")
    print(f"Std reward: {np.std(stats['rewards']):.3f}")
    print(f"Min reward: {np.min(stats['rewards']):.3f}")
    print(f"Max reward: {np.max(stats['rewards']):.3f}")
    print(f"Mean action magnitude: {np.mean(stats['action_magnitudes']):.3f}")
    print(f"Mean action change: {np.mean(stats['action_changes']):.3f}")

    growth_info = agent.get_growth_info()
    print(f"\n=== Action Space Growth Summary ===")
    print(f"Final resolution: {growth_info['current_bins']} bins per dimension")
    print(f"Growth history: {growth_info['growth_history']}")
    print(f"Maximum possible resolution: {growth_info['max_bins']} bins")


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
        "--video-path", type=str, default=None, help="Path to save video"
    )
    parser.add_argument(
        "--no-display", action="store_true", help="Don't show video display window"
    )
    parser.add_argument("--fps", type=int, default=30, help="Video frame rate")

    args = parser.parse_args()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )

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