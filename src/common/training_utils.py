import os
from pathlib import Path

import torch
import torch.nn.functional as F
from dm_control import suite


def random_shift(images, pad_size=4):
    """Apply random shift augmentation to images."""
    n, c, h, w = images.shape
    padded = F.pad(images, [pad_size] * 4, mode="replicate")
    top = torch.randint(0, 2 * pad_size + 1, (n,))
    left = torch.randint(0, 2 * pad_size + 1, (n,))

    shifted_images = torch.zeros_like(images)
    for i in range(n):
        shifted_images[i] = padded[i, :, top[i] : top[i] + h, left[i] : left[i] + w]

    return shifted_images


def get_obs_shape(use_pixels: bool, obs_spec: dict) -> tuple:
    """Get observation shape."""
    if use_pixels:
        return (3, 84, 84)
    else:
        state_dim = sum(
            spec.shape[0] if len(spec.shape) > 0 else 1 for spec in obs_spec.values()
        )
        return (state_dim,)


def process_pixel_observation(dm_obs, device):
    if "pixels" in dm_obs:
        obs = dm_obs["pixels"]
    else:
        camera_obs = [v for k, v in dm_obs.items() if "camera" in k or "rgb" in k]
        if camera_obs:
            obs = camera_obs[0]
        else:
            raise ValueError("No pixel observations found in DM Control observation")

    obs = torch.tensor(obs, dtype=torch.float32, device=device)
    if len(obs.shape) == 3:  # HWC -> CHW
        obs = obs.permute(2, 0, 1)
    return obs


def process_observation(dm_obs, use_pixels, device, obs_buffer=None) -> torch.Tensor:
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
        if obs_buffer is None:
            return torch.from_numpy(state_vector).to(device)
        else:
            return obs_buffer.update(torch.from_numpy(state_vector))


def apply_action_penalty(rewards, actions, penalty_coeff):
    """
    Apply action penalty as used in paper experiments.
    """
    if penalty_coeff <= 0:
        return rewards

    if isinstance(actions, torch.Tensor):
        actions = actions.cpu().numpy()
    actions = np.array(actions)
    if len(actions.shape) == 1:
        action_norm_squared = np.sum(actions**2)
        M = len(actions)
    else:
        action_norm_squared = np.sum(actions**2, axis=-1)
        M = actions.shape[-1] if len(actions.shape) > 1 else 1

    # Penalty = ca * ||a||² / M, where ca is penalty coefficient, M is action dimension
    penalties = penalty_coeff * action_norm_squared / M
    penalties = np.clip(penalties, 0, abs(rewards) * 0.1)

    return rewards - penalties


def get_path(base_path, filename="", middle_path="", logger=None, create=False):
    path = Path(__file__).parents[2] / base_path / middle_path / filename
    if create:
        path.parent.mkdir(exist_ok=True, parents=True)
        if logger:
            logger.info(f"Created directory {path}")
    return path


import torch
import numpy as np


def huber_loss(
    td_error: torch.Tensor, huber_loss_parameter: float = 1.0
) -> torch.Tensor:
    abs_error = torch.abs(td_error)
    quadratic = torch.minimum(
        abs_error, torch.tensor(huber_loss_parameter, device=abs_error.device)
    )
    linear = abs_error - quadratic
    return 0.5 * quadratic**2 + huber_loss_parameter * linear


def get_combined_random_and_greedy_actions(
    q_max, num_dims, num_bins, batch_size, epsilon, device
):
    random_mask = torch.rand(batch_size, num_dims, device=device) < epsilon
    random_actions = torch.randint(0, num_bins, (batch_size, num_dims), device=device)
    greedy_actions = q_max.argmax(dim=2)
    actions = torch.where(random_mask, random_actions, greedy_actions)
    return actions


def continuous_to_discrete_action(
    config, action_discretizer, continuous_action: torch.Tensor
) -> np.ndarray:
    if isinstance(continuous_action, torch.Tensor):
        continuous_action = continuous_action.cpu().numpy()

    continuous_action = np.array(continuous_action)

    if config.decouple:
        discrete_action = []
        for dim in range(len(continuous_action)):
            bins = action_discretizer.action_bins[dim].cpu().numpy()
            closest_idx = np.argmin(np.abs(bins - continuous_action[dim]))
            discrete_action.append(closest_idx)
        return np.array(discrete_action, dtype=np.int64)
    else:
        action_bins_cpu = action_discretizer.action_bins.cpu().numpy()
        distances = np.linalg.norm(action_bins_cpu - continuous_action, axis=1)
        return np.argmin(distances)


def get_env_specs(env, use_pixels=False):
    action_spec = env.action_spec()
    obs_spec = env.observation_spec()
    obs_shape = get_obs_shape(use_pixels, obs_spec)
    action_spec_dict = {"low": action_spec.minimum, "high": action_spec.maximum}
    return obs_shape, action_spec_dict


def check_task(task, logger):
    if task not in [f"{domain}_{task}" for domain, task in suite.ALL_TASKS]:
        logger.warn(f"Task {task} not found, using walker_walk")
        task = "walker_walk"
    return task


def get_env(task, logger):
    task = check_task(task, logger)
    domain_name, task_name = task.split("_", 1)
    env = suite.load(domain_name, task_name)
    return env


def init_training(seed, device, logger):
    os.makedirs("output/checkpoints", exist_ok=True)
    os.makedirs("output/metrics", exist_ok=True)
    os.makedirs("output/plots", exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    logger.info(f"Using device: {device}")
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True


def check_and_sample_batch_from_replay_buffer(
    replay_buffer, min_replay_size, batch_size
):
    if len(replay_buffer) < min_replay_size:
        return None
    return replay_buffer.sample(batch_size)


def get_batch_components(batch, device):
    obs, actions, rewards, next_obs, dones, discounts, weights, indices = batch
    obs = obs.to(device)
    actions = actions.to(device)
    rewards = rewards.to(device)
    next_obs = next_obs.to(device)
    dones = dones.to(device)
    discounts = discounts.to(device)
    weights = weights.to(device)
    return obs, actions, rewards, next_obs, dones, discounts, weights, indices


def encode_observation(encoder, obs, next_obs):
    if encoder:
        obs_encoded = encoder(obs)
        with torch.no_grad():
            next_obs_encoded = encoder(next_obs)
    else:
        obs_encoded = obs.flatten(1)
        next_obs_encoded = next_obs.flatten(1)
    return obs_encoded, next_obs_encoded


def calculate_losses(
    td_error1,
    td_error2,
    use_double_q,
    q_optimizer,
    encoder,
    encoder_optimizer,
    weights,
    huber_loss_parameter,
):
    loss1 = huber_loss(td_error1, huber_loss_parameter)
    loss2 = (
        huber_loss(td_error2, huber_loss_parameter)
        if use_double_q
        else torch.zeros_like(loss1)
    )
    loss1 = (loss1 * weights).mean()
    loss2 = (loss2 * weights).mean() if use_double_q else torch.zeros_like(loss1)

    total_loss = loss1 + loss2

    q_optimizer.zero_grad()
    if encoder:
        encoder_optimizer.zero_grad()

    total_loss.backward()
    return total_loss
