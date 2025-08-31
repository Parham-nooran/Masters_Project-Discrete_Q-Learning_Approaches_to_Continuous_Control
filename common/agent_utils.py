import torch
import numpy as np

def huber_loss(td_error: torch.Tensor, huber_loss_parameter: float = 1.0) -> torch.Tensor:
    abs_error = torch.abs(td_error)
    quadratic = torch.minimum(
        abs_error, torch.tensor(huber_loss_parameter, device=abs_error.device)
    )
    linear = abs_error - quadratic
    return 0.5 * quadratic ** 2 + huber_loss_parameter * linear

def get_combined_random_and_greedy_actions(q_max, num_dims, num_bins, batch_size, epsilon, device):
    random_mask = torch.rand(batch_size, num_dims, device=device) < epsilon

    # Random actions for exploration
    random_actions = torch.randint(0, num_bins, (batch_size, num_dims), device=device)

    # Greedy actions for exploitation
    greedy_actions = q_max.argmax(dim=2)

    # Combine using the mask
    actions = torch.where(random_mask, random_actions, greedy_actions)
    return actions


def continuous_to_discrete_action(config, action_discretizer, continuous_action: torch.Tensor) -> np.ndarray:
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