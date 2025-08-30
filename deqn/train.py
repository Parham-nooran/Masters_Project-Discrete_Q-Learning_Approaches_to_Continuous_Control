import time
import gc
import torch
from config import *

from dec_qn_agent import DecQNAgent
from plotting_utils import *
from dm_control import suite
import numpy as np


def find_latest_checkpoint(checkpoint_dir="checkpoints"):
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    if not checkpoint_files:
        return None

    checkpoint_files.sort(
        key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True
    )
    latest_file = checkpoint_files[0]

    return os.path.join(checkpoint_dir, latest_file)


def save_checkpoint(agent, episode, path):

    checkpoint = {
        "episode": episode,
        "q_network_state_dict": agent.q_network.state_dict(),
        "target_q_network_state_dict": agent.target_q_network.state_dict(),
        "q_optimizer_state_dict": agent.q_optimizer.state_dict(),
        "config": agent.config,
        "training_step": agent.training_step,
        "epsilon": agent.epsilon,
        "replay_buffer_buffer": agent.replay_buffer.buffer,
        "replay_buffer_position": agent.replay_buffer.position,
        "replay_buffer_priorities": agent.replay_buffer.priorities,
        "replay_buffer_max_priority": agent.replay_buffer.max_priority,
    }

    if agent.encoder:
        checkpoint["encoder_state_dict"] = agent.encoder.state_dict()
        checkpoint["encoder_optimizer_state_dict"] = (
            agent.encoder_optimizer.state_dict()
        )

    torch.save(checkpoint, path)


def process_pixel_observation(dm_obs, device):
    if "pixels" in dm_obs:
        obs = dm_obs["pixels"]
    else:
        camera_obs = [v for k, v in dm_obs.items() if "camera" in k or "rgb" in k]
        if camera_obs:
            obs = camera_obs[0]
        else:
            raise ValueError(
                "No pixel observations found in DM Control observation"
            )

    obs = torch.tensor(obs, dtype=torch.float32, device=device)
    if len(obs.shape) == 3:  # HWC -> CHW
        obs = obs.permute(2, 0, 1)

    return obs

def process_observation(dm_obs, use_pixels, device):
    if use_pixels:
        return process_pixel_observation(dm_obs, device)
    else:
        state_parts = []
        for key in sorted(dm_obs.keys()):
            val = dm_obs[key]
            if isinstance(val, np.ndarray):
                state_parts.append(val.astype(np.float32).flatten())
            else:
                state_parts.append(np.array([float(val)], dtype=np.float32))

        state_vector = np.concatenate(state_parts, dtype=np.float32)
        return torch.from_numpy(state_vector).to(device)


def handle_checkpoint_loading(agent, checkpoint_to_load):
    start_episode = 0
    if checkpoint_to_load is None:
        latest_checkpoint = find_latest_checkpoint()
        if latest_checkpoint:
            checkpoint_to_load = latest_checkpoint
            print(f"Found latest checkpoints: {latest_checkpoint}")

    if checkpoint_to_load:
        if os.path.exists(checkpoint_to_load):
            try:
                loaded_episode = agent.load_checkpoint(checkpoint_to_load)
                start_episode = loaded_episode + 1
                print(
                    f"Resumed from episode {loaded_episode}, starting at episode {start_episode}"
                )
            except Exception as e:
                print(f"Failed to load checkpoints: {e}")
                print("Starting fresh training...")
                start_episode = 0
        else:
            print(
                f"Checkpoint file {checkpoint_to_load} not found. Starting fresh training..."
            )
    return start_episode

def get_obs_shape(use_pixels, obs_spec):
    if use_pixels:
        obs_shape = (3, 84, 84)  # RGB camera view
    else:
        state_dim = sum(
            spec.shape[0] if len(spec.shape) > 0 else 1 for spec in obs_spec.values()
        )
        obs_shape = (state_dim,)
    return obs_shape

def get_numpy_action(action):
    if isinstance(action, torch.Tensor):
        action_np = action.cpu().numpy()
    else:
        action_np = action
    return action_np

def optimize_gpu_usage(device):
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True


def train_decqn():
    args = parse_args()
    config = create_config_from_args(args)

    # Device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    optimize_gpu_usage(device)

    if config.task not in [f"{domain}_{task}" for domain, task in suite.ALL_TASKS]:
        print(f"Warning: Task {config.task} not found in suite, using walker_walk")
        config.task = "walker_walk"

    domain_name, task_name = config.task.split("_", 1)
    env = suite.load(domain_name, task_name)
    action_spec = env.action_spec()
    obs_spec = env.observation_spec()
    obs_shape = get_obs_shape(config.use_pixels, obs_spec)
    action_spec_dict = {"low": action_spec.minimum, "high": action_spec.maximum}
    agent = DecQNAgent(config, obs_shape, action_spec_dict)
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_to_load = args.load_checkpoints
    start_episode = handle_checkpoint_loading(agent, checkpoint_to_load)
    metrics_tracker = MetricsTracker(save_dir="metrics")
    if start_episode > 0:
        metrics_tracker.load_metrics()
    print(f"Agent setup - decouple: {agent.config.decouple}, action_dim: {agent.action_discretizer.action_dim}")
    print("Starting training...")
    start_time = time.time()

    for episode in range(start_episode, config.num_episodes):
        episode_start_time = time.time()
        episode_reward = 0
        episode_loss = 0
        episode_q_mean = 0
        loss_count = 0

        time_step = env.reset()
        obs = process_observation(time_step.observation, config.use_pixels, device)
        agent.observe_first(obs)

        step = 0
        while not time_step.last():
            action = agent.select_action(obs)
            action_np = get_numpy_action(action)
            time_step = env.step(action_np)
            next_obs = process_observation(time_step.observation, config.use_pixels, device)
            reward = time_step.reward if time_step.reward is not None else 0.0
            done = time_step.last()
            agent.observe(action, reward, next_obs, done)

            # Update agent
            if len(agent.replay_buffer) > config.min_replay_size:
                metrics = agent.update()
                if metrics:
                    if "loss" in metrics and metrics["loss"] is not None:
                        episode_loss += metrics["loss"]
                    episode_q_mean += metrics.get("q1_mean", 0)
                    loss_count += 1

            obs = next_obs
            episode_reward += reward
            step += 1

            if step > 1000:
                break

        # Calculate averages
        avg_loss = episode_loss / max(loss_count, 1) if loss_count > 0 else 0.0
        avg_q_mean = episode_q_mean / max(loss_count, 1)

        # Log metrics
        metrics_tracker.log_episode(
            episode=episode,
            reward=episode_reward,
            length=step,
            loss=avg_loss if loss_count > 0 else None,
            q_mean=avg_q_mean if loss_count > 0 else None,
            epsilon=agent.epsilon,
        )

        agent.update_epsilon(decay_rate=0.995, min_epsilon=0.01)

        if device == "cuda":
            torch.cuda.synchronize()

        # Enhanced progress logging
        episode_time = time.time() - episode_start_time
        if episode % args.log_interval == 0:
            print(
                f"Episode {episode:4d} | "
                f"Reward: {episode_reward:7.2f} | "
                f"Loss: {avg_loss:8.6f} | "
                f"Q-mean: {avg_q_mean:6.3f} | "
                f"Time: {episode_time:.2f}s | "
                f"Buffer: {len(agent.replay_buffer):6d}"
            )
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
                f"Recent {args.detailed_log_interval} episodes avg reward: {np.mean(metrics_tracker.episode_rewards[-args.detailed_log_interval:]):.2f}"
            )
            print(
                f"Elapsed Time: {elapsed_time / 60:.1f} min | ETA: {eta / 60:.1f} min"
            )
            print("-" * 35)

        # Clear GPU cache periodically
        if episode % 10 == 0 and device == "cuda":
            torch.cuda.empty_cache()

        # Save checkpoints
        if episode % args.checkpoint_interval == 0:
            import shutil

            # Save metrics before clearing checkpoints
            metrics_tracker.save_metrics()

            # Remove old checkpoints directory and create new one
            if os.path.exists("checkpoints"):
                shutil.rmtree("checkpoints")
            os.makedirs("checkpoints", exist_ok=True)

            checkpoint_path = f"./checkpoints/decqn_episode_{episode}.pth"
            save_checkpoint(agent, episode, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    # Save final checkpoints
    import shutil

    metrics_tracker.save_metrics()  # Save metrics first
    if os.path.exists("checkpoints"):
        shutil.rmtree("checkpoints")
    os.makedirs("checkpoints", exist_ok=True)
    final_checkpoint = "checkpoints/decqn_final.pth"
    save_checkpoint(agent, config.num_episodes, final_checkpoint)
    print(f"Final checkpoints saved: {final_checkpoint}")

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


if __name__ == "__main__":
    agent = train_decqn()
