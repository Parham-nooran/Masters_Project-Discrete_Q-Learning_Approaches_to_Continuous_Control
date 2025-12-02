"""Utilities for observation processing."""

import numpy as np
import torch


def process_pixel_observation(dm_obs, device):
    """Process pixel observation from DM Control."""
    if "pixels" in dm_obs:
        obs = dm_obs["pixels"]
    else:
        camera_obs = [v for k, v in dm_obs.items() if "camera" in k or "rgb" in k]
        if camera_obs:
            obs = camera_obs[0]
        else:
            raise ValueError("No pixel observations found")

    obs = torch.tensor(obs, dtype=torch.float32, device=device)
    if len(obs.shape) == 3:
        obs = obs.permute(2, 0, 1)
    return obs


def process_state_observation(dm_obs):
    """Process state-based observation from DM Control."""
    if isinstance(dm_obs, np.ndarray):
        return dm_obs.astype(np.float32).flatten()

    state_parts = []
    for key in sorted(dm_obs.keys()):
        val = dm_obs[key]
        if isinstance(val, np.ndarray):
            state_parts.append(val.astype(np.float32).flatten())
        else:
            state_parts.append(np.array([float(val)], dtype=np.float32))
    return np.concatenate(state_parts, dtype=np.float32)


def process_observation(dm_obs, use_pixels, device, obs_buffer=None, env_type="dmcontrol") -> torch.Tensor:
    """Process observation based on type and environment.

    Args:
        dm_obs: Observation from environment
        use_pixels: Whether using pixel observations
        device: Torch device
        obs_buffer: Optional observation buffer
        env_type: Type of environment ('dmcontrol' or 'ogbench')
    """
    if env_type == "ogbench":
        if use_pixels:
            obs = torch.tensor(dm_obs, dtype=torch.float32, device=device)
            if len(obs.shape) == 3 and obs.shape[-1] == 3:
                obs = obs.permute(2, 0, 1)  # HWC to CHW
            return obs
        else:
            obs = torch.from_numpy(dm_obs.astype(np.float32)).to(device)
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


def encode_observation(encoder, obs, next_obs):
    """Encode observations using encoder or flatten."""
    if encoder:
        obs_encoded = encoder(obs)
        with torch.no_grad():
            next_obs_encoded = encoder(next_obs)
    else:
        obs_encoded = obs.flatten(1)
        next_obs_encoded = next_obs.flatten(1)
    return obs_encoded, next_obs_encoded
