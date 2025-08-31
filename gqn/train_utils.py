import torch
import numpy as np


def process_observation(dm_obs, use_pixels, device):
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
        return torch.from_numpy(state_vector).to(device)
