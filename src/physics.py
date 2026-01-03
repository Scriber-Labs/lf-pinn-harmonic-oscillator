from __future__ import annotations

from typing import Final, Protocol

import torch
from torch import Tensor, autograd

__all__: list[str] = [
    "ho_variational_loss"
]  # functions imported via 'from physics import *'


# ------------------------------------------------------------------------------
# 0️⃣ Typing helpers
# ------------------------------------------------------------------------------
class SupportsForward(Protocol):
    """
    Anything that can be called like a neural network model: ``x = model(t)``.
    Both `torch.nn.Module` and plain callablables satisfy the protocol.
    """

    def __call__(self, t: Tensor) -> Tensor:  # noqa: D401
        ...

# ------------------------------------------------------------------------------
# 2️⃣ Public API
# ------------------------------------------------------------------------------
def ho_variational_loss(
    model: SupportsForward,
    t: Tensor,
    *,
    omega: float | Tensor = 1.0,
    reduction: str = "mean",
) -> Tensor:
    """
    Variational loss for the simple harmonic oscillator (SHO).

    Enforces the ode ``ddot(x) + omega**2 * x(t) = 0`` as a soft constraint by penalizing the squared residual at the collacation points `t`.

    Parameters
    ----------
    model :
        collable that maps a tensor of time coordinates ``t`` to the predicted displacement ``x(t)``. Usually an `nn.Module`.
    t :
        1-D or 2-D tensor of time coordinates with `required_grad=False`. Its dtype / device dictate those of the intermediate computations.
    omega :
        Natural angular frequency of the oscillator. Can be a float or a 0-D tensor.
    reduction :
        Specifies the reduction to apply to the point-wise residuals:
          - "mean" -> `torch.mean` (default)
          - "sum"  -> `torch.sum`
          - "none" -> no reduction, returns the unreduced residual per sample

    Returns
    -------
    torch.Tensor
        Scalar tensor if reduction os "mean" or "sum", else a 1-D tensor of shape ``(len(t),)`` containing the point-wise squared residuals.

    Notes
    -----
    - `create_graph=True` is required so that higher-order derivatives remain part of teh autograd graph (this is useful for optimization steps involving second derivatives).
    - No 'torch.no_grad` context should enclose this call.
    """
    if not torch.is_tensor(t):
        raise TypeError(f"`t` must be a torch.Tensor, got {type(t)}")

    # Ensure `t` participates in autograd.
    t = t.detach().requires_grad_(True)

    # Forward pass: x = x(t)
    x: Tensor = model(t)

    # First and second derivatives:
    dx: Tensor = _grad(x, t)
    ddx: Tensor = _grad(dx, t)

    # Residuals of the SHO ODE collocation points
    residual: Tensor = ddx + (omega**2) * x
    pointwise_loss: Tensor = residual.pow(2)  # squared residuals

    if reduction == "mean":
        return pointwise_loss.mean()
    elif reduction == "sum":
        return pointwise_loss.sum()
    elif reduction == "none":
        return pointwise_loss.squeeze()
    else:
        raise ValueError(
            "reduction must be one of 'mean', 'sum', or 'none'," f" got '{reduction}'."
        )


# ------------------------------------------------------------------------------
# 3️⃣ Private helpers
# ------------------------------------------------------------------------------
def _grad(y: Tensor, x: Tensor) -> Tensor:
    """
    Convenience wrapper around `torch.autograd.grad` that returns the first element of the gradient tuple and keeps the graph.

    Equivalent to:

      ```python
      torch.autograd.grad(
          y, x,
          grad_output=torch.ones_like(y),
          create_graph=True
      )[0]
      ```
    """
    (dy_dx,) = autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
    )
    return dy_dx


# ------------------------------------------------------------------------------
# 4️⃣ Example usage/ test case (run `python -m physics` from project root)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    from model import MLP

    torch.manual_seed(0)

    device: Final = torch.device("cpu")
    net = MLP(hidden=64, device=device)

    # 20 random collocation points in the interval [0, 2pi]
    t_collocation: Tensor = 2 * torch.pi * torch.rand(20, 1, device=device)
    loss: Tensor = ho_variational_loss(net, t_collocation, omega=1.0)
    print(f"Variational loss: {loss.item():.6f}")
