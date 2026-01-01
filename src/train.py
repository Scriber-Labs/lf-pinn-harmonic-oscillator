from __future__ import annotations 

import argparse
import random
import sys
from pathlib import Path 
from typing import Final

import numpy as np
import torch
from torch import Tensor, nn, optim

# Local imports
from model import MLP 
from physics import ho_variational_loss
from utils import set_global_seed

__all__: list[str] = ["main", "train", "make_collocoation_points"]

# ------------------------------------------------------------------------------
# 0️⃣ Data helpers
# ------------------------------------------------------------------------------
def make_collocation_points(
    n_points: int,
    *, 
    t_max: float = 2 * np.pi,
    device: torch.device | str = "cpu",
) -> Tensor:
    """
    Draw random time coordinates in ``[0, t_max]`` for collocation points.
    
    Returns
    -------
    torch.Tensor
        Shape ``(n_points, 1)`` with dtype ``torch.float32`` on ``device``.
    """
    device = torch.device(device)
    return t_max * torch.rand(n_points, 1, device=device)

# ------------------------------------------------------------------------------
# 1️⃣ Main training loop
# ------------------------------------------------------------------------------
def train(
    model: nn.Module,
    collocation: Tensor,
    *,
    omega: float = 1.0,
    epochs: int = 3_000,
    lr: float = 1e-3,
    log_every: int = 500,
) -> list[float]:
    """
    Simple training loop for the 1-D simple harmonic oscillator PINN.
    
    Parameters
    ----------
    model : nn.Module
        The neural network (e.g., `MLP`).
    collocation : torch.Tensor
        Time coordinates where the ODE residual is enforced. Shape ``(n_points, 1)``.
    omega : float, default = 1.0
        Natural angular frequency of the oscillator.
    epochs : int, default = 3_000
        Number of gradient descent steps.
    lr : float, default = 1e-3
        Learning rate for the Adam optimizer.
    log_every : int, default = 500
        Print progress every 'log_every' epochs.
        
    Returns
    -------
    list[float]
        Loss history (one entry per epoch).
    """
    optimizer = optim.Optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_history: list[float] = []
    
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        loss = Tensor = ho_variational_loss(model, collocation, omega=omega)
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if epoch % log_every == 0 or epoch == 1:
            print(f"epoch {epoch:5d} | loss = {loss.item():.6e}")
            
    return loss_history

# ------------------------------------------------------------------------------
# 2️⃣ CLI
# ------------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m trian",
        description="Train a low-fidelity PINN for the simple harmonic oscillator.",
    )
    parser.add_argument("--hidden", type=int, default=64, help="number of hidden units per layer")
    parser.add_argument("--epochs", type=int, default=3000, help="training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--n-points", type=int, default=200, help="collocation points")
    parser.add_argument("--omega", type=float, default=1.0, help="angular frequency")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cpu | cuda | cuda:0 | ...",
    )
    parser.add_argument("--log-every", type=int, default=500, help="print interval")
    return parser.parse_args(argv)

# ------------------------------------------------------------------------------
# 3️⃣ Entry-point
# ------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:    # noqa: D401
    """
    CLI entry-point so that ``python -m train`` or ``pinn-train`` works.
    """
    args = parse_args(argv)
    
    set_global_seed(args.seed, deterministic=True)
    
    device: Final = torch.device(args.device)
    print("Using device:", device)
    
    # Model and data
    model = MLP(hidden=args.hidden, device=device)
    collocation = make_collocation_points(
        args.n_points, device=device, t_max=2 * np.pi
    )
    
    # Training
    print(
        f"Starting training: epochs={args.epochs}, "
        f"lr={args.lr}, collocation={len(collocation)}"
    )
    loss_history = train(
        model,
        collocation,
        omega=args.omega,
        epochs=args.epochs,
        lr=args.lr,
        log_every=args.log_every,
    )
    
    print("Training finished. Final loss:", f"{loss_history[-1]:.6e}")
    

# ------------------------------------------------------------------------------
# 4️⃣ Run if executed as a script
# ------------------------------------------------------------------------------ 
if __name__ == "__main__":
    main(sys.argv[1:])
