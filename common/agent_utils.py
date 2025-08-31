import torch

def huber_loss(td_error: torch.Tensor, huber_loss_parameter: float = 1.0) -> torch.Tensor:
    abs_error = torch.abs(td_error)
    quadratic = torch.minimum(
        abs_error, torch.tensor(huber_loss_parameter, device=abs_error.device)
    )
    linear = abs_error - quadratic
    return 0.5 * quadratic ** 2 + huber_loss_parameter * linear

