"""
Utility functions for coarse-to-fine action discretization.

These functions handle encoding/decoding between continuous and discrete
action representations and zooming into refined action ranges.
"""

import torch


def random_action_if_within_delta(qs: torch.Tensor, delta: float = 0.0001) -> torch.Tensor:
    """
    Return random action if Q-values are too similar.

    Helps with exploration when multiple actions have similar values.

    Args:
        qs: Q-values [B, action_dim, bins].
        delta: Threshold for considering Q-values as equal.

    Returns:
        Action indices, or None if no random selection needed.
    """
    q_diff = qs.max(-1).values - qs.min(-1).values
    random_action_mask = q_diff < delta
    if random_action_mask.sum() == 0:
        return None
    argmax_q = qs.max(-1)[1]
    random_actions = torch.randint(0, qs.size(-1), random_action_mask.shape).to(
        qs.device
    )
    argmax_q = torch.where(random_action_mask, random_actions, argmax_q)
    return argmax_q


def encode_action(
        continuous_action: torch.Tensor,
        initial_low: torch.Tensor,
        initial_high: torch.Tensor,
        levels: int,
        bins: int,
) -> torch.Tensor:
    """
    Encode continuous action to hierarchical discrete representation.

    Progressively refines action discretization across levels by zooming
    into narrower ranges.

    Args:
        continuous_action: Continuous actions [..., action_dim].
        initial_low: Lower bounds of action space [action_dim].
        initial_high: Upper bounds of action space [action_dim].
        levels: Number of hierarchy levels.
        bins: Number of bins per level.

    Returns:
        Discrete actions [..., levels, action_dim].
    """
    low = initial_low.repeat(*continuous_action.shape[:-1], 1)
    high = initial_high.repeat(*continuous_action.shape[:-1], 1)

    idxs = []
    for _ in range(levels):
        slice_range = (high - low) / bins
        idx = torch.floor((continuous_action - low) / slice_range)
        idx = torch.clip(idx, 0, bins - 1)
        idxs.append(idx)

        recalculated_action = low + slice_range * idx
        recalculated_action = torch.clip(recalculated_action, -1.0, 1.0)
        low = recalculated_action
        high = recalculated_action + slice_range
        low = torch.maximum(-torch.ones_like(low), low)
        high = torch.minimum(torch.ones_like(high), high)

    discrete_action = torch.stack(idxs, -2)
    return discrete_action


def decode_action(
        discrete_action: torch.Tensor,
        initial_low: torch.Tensor,
        initial_high: torch.Tensor,
        levels: int,
        bins: int,
) -> torch.Tensor:
    """
    Decode hierarchical discrete action to continuous representation.

    Reconstructs continuous action by following the hierarchical
    discretization process.

    Args:
        discrete_action: Discrete actions [..., levels, action_dim].
        initial_low: Lower bounds of action space [action_dim].
        initial_high: Upper bounds of action space [action_dim].
        levels: Number of hierarchy levels.
        bins: Number of bins per level.

    Returns:
        Continuous actions [..., action_dim].
    """
    low = initial_low.repeat(*discrete_action.shape[:-2], 1)
    high = initial_high.repeat(*discrete_action.shape[:-2], 1)

    for i in range(levels):
        slice_range = (high - low) / bins
        continuous_action = low + slice_range * discrete_action[..., i, :]
        low = continuous_action
        high = continuous_action + slice_range
        low = torch.maximum(-torch.ones_like(low), low)
        high = torch.minimum(torch.ones_like(high), high)

    continuous_action = (high + low) / 2.0
    return continuous_action


def zoom_in(
        low: torch.Tensor,
        high: torch.Tensor,
        argmax_q: torch.Tensor,
        bins: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Zoom into the selected bin's range for next level.

    Takes the current range and narrows it to the selected bin,
    which becomes the range for the next hierarchy level.

    Args:
        low: Current lower bounds [B, action_dim].
        high: Current upper bounds [B, action_dim].
        argmax_q: Selected bin indices [B, action_dim].
        bins: Number of bins.

    Returns:
        Tuple of (new_low, new_high) for next level.
    """
    slice_range = (high - low) / bins
    continuous_action = low + slice_range * argmax_q
    low = continuous_action
    high = continuous_action + slice_range
    low = torch.maximum(-torch.ones_like(low), low)
    high = torch.minimum(torch.ones_like(high), high)
    return low, high