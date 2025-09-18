import torch
import torch.nn.functional as F
import numpy as np


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
