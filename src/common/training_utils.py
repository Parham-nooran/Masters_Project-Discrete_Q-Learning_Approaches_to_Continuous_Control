import os
from pathlib import Path

import numpy as np
import torch
from dm_control import suite

from src.common.env_factory import create_dmcontrol_env, create_ogbench_env, create_metaworld_env
from src.common.observation_utils import process_state_observation


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


def process_observation(dm_obs, use_pixels, device, obs_buffer=None, env_type="dmcontrol") -> torch.Tensor:
    """Process observation based on type and environment.

    Args:
        dm_obs: Observation from environment
        use_pixels: Whether using pixel observations
        device: Torch device
        obs_buffer: Optional observation buffer
        env_type: Type of environment ('dmcontrol' or 'ogbench')
    """
    if env_type in ["ogbench", "metaworld"]:
        if use_pixels:
            obs = torch.tensor(dm_obs, dtype=torch.float32, device=device)
            if len(obs.shape) == 3 and obs.shape[-1] == 3:
                obs = obs.permute(2, 0, 1)
            return obs
        else:
            if isinstance(dm_obs, np.ndarray):
                obs = torch.from_numpy(dm_obs.astype(np.float32)).to(device)
            else:
                obs = torch.tensor(dm_obs, dtype=torch.float32, device=device)
            if obs_buffer is None:
                return obs
            return obs_buffer.update(obs)
    else:
        if use_pixels:
            return process_pixel_observation(dm_obs, device)

        state_vector = process_state_observation(dm_obs)
        if obs_buffer is None:
            return torch.from_numpy(state_vector).to(device)
        return obs_buffer.update(torch.from_numpy(state_vector))


def get_path(base_path, filename="", middle_path="", logger=None, create=False):
    path = Path(__file__).parents[2] / base_path / middle_path / filename
    if create:
        path.parent.mkdir(exist_ok=True, parents=True)
        if logger:
            logger.info(f"Created directory {path}")
    return path


def huber_loss(
        td_error: torch.Tensor, huber_loss_parameter: float = 1.0
) -> torch.Tensor:
    abs_error = torch.abs(td_error)
    quadratic = torch.minimum(
        abs_error, torch.tensor(huber_loss_parameter, device=abs_error.device)
    )
    linear = abs_error - quadratic
    return 0.5 * quadratic ** 2 + huber_loss_parameter * linear


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

def check_task(task, logger):
    if task not in [f"{domain}_{task}" for domain, task in suite.ALL_TASKS]:
        logger.warn(f"Task {task} not found, using walker_walk")
        task = "walker_walk"
    return task


def get_env(task, logger, seed, env_type="dmcontrol"):
    """Get environment based on type.

    Args:
        task: Task name
        logger: Logger instance
        env_type: Type of environment ('dmcontrol' or 'ogbench')

    Returns:
        Environment instance
    """

    if env_type == "ogbench":
        logger.info(f"Creating OGBench environment: {task}")
        env = create_ogbench_env(task, seed)
    elif env_type == "metaworld":
        logger.info(f"Creating metaworld environment: {task}")
        env = create_metaworld_env(task, seed)
    else:
        domain_name, task_name = task.split("_", 1)
        logger.info(f"Creating dm_control environment: {domain_name}/{task_name}")
        env = create_dmcontrol_env(domain_name, task_name, seed)
    return env


def get_env_specs(env, use_pixels, env_type="dmcontrol"):
    """Get observation shape and action spec from environment.

    Args:
        env: Environment instance
        use_pixels: Whether using pixel observations
        env_type: Type of environment

    Returns:
        Tuple of (obs_shape, action_spec_dict)
    """
    if env_type == "ogbench":
        obs_space = env.env.observation_space
        action_space = env.env.action_space

        if use_pixels:
            obs_shape = obs_space.shape  # Should be (C, H, W) or (H, W, C)
        else:
            obs_shape = (int(obs_space.shape[0]),)

        action_spec_dict = {
            "low": action_space.low,
            "high": action_space.high,
            "shape": action_space.shape
        }
    else:
        action_spec = env.action_spec()
        obs_spec = env.observation_spec()

        if use_pixels:
            obs_shape = next(iter(obs_spec.values())).shape
        else:
            total_size = sum(int(np.prod(spec.shape)) for spec in obs_spec.values())
            obs_shape = (total_size,)

        action_spec_dict = {
            "low": action_spec.minimum,
            "high": action_spec.maximum,
            "shape": action_spec.shape
        }

    return obs_shape, action_spec_dict


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
