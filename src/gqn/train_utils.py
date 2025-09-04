import torch
import numpy as np


def process_observation(dm_obs, use_pixels, device, obs_buffer=None):
    """Process DeepMind Control observations."""
    if use_pixels:
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
    else:
        # State-based observation processing
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
    FIXED: Proper penalty calculation and scaling.
    """
    if penalty_coeff <= 0:
        return rewards

    if isinstance(actions, torch.Tensor):
        actions = actions.cpu().numpy()

    # Ensure actions is numpy array
    actions = np.array(actions)

    # Calculate L2 norm of actions
    if len(actions.shape) == 1:
        action_norm_squared = np.sum(actions ** 2)
        M = len(actions)
    else:
        action_norm_squared = np.sum(actions ** 2, axis=-1)
        M = actions.shape[-1] if len(actions.shape) > 1 else 1

    # FIXED: Proper penalty calculation from paper
    # Penalty = ca * ||a||² / M, where ca is penalty coefficient, M is action dimension
    penalties = penalty_coeff * action_norm_squared / M

    # FIXED: Ensure penalty doesn't dominate the reward signal
    penalties = np.clip(penalties, 0, abs(rewards) * 0.1)  # Limit penalty to 10% of reward magnitude

    return rewards - penalties


class OptimizedObsBuffer:
    """Optimized observation buffer for tensor operations."""

    def __init__(self, obs_shape, device):
        self.device = device
        if len(obs_shape) == 1:  # State-based
            self.obs_buffer = torch.zeros(obs_shape, dtype=torch.float32, device=device)
        else:  # Pixel-based
            self.obs_buffer = torch.zeros(obs_shape, dtype=torch.float32, device=device)

    def update(self, new_obs):
        """Update buffer with new observation."""
        if isinstance(new_obs, np.ndarray):
            self.obs_buffer.copy_(torch.from_numpy(new_obs))
        else:
            self.obs_buffer.copy_(new_obs)
        return self.obs_buffer