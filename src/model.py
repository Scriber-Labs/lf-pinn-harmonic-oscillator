from __future__ import annotations

from typing import Final

import torch
from torch import Tensor, nn

from utils import set_global_seed

# ------------------------------------------------------------------------------
# 1️⃣ Neural network model
# ------------------------------------------------------------------------------
class MLP(nn.Module):
    """
    A minimal 3-layer multilayer perceptron.

    Parameters
    ----------
    hidden : int, default = 64
        Number of hidden units per hidden layer.
    device : torch.device or st r, optional
        Device on which the parameters will be allocated. If omitted, they will be kept on the current default device.
    dtype : torch.dtype, optional
        Floating-point precision of the parameters.
    """

    INPUT_DIM: Final[int] = 1
    OUTPUT_DIM: Final[int] = 1

    def __init__(
        self,
        hidden: int = 64,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        # Build the network
        self.net: nn.Sequential = nn.Sequential(
            nn.Linear(self.INPUT_DIM, hidden, device=device, dtype=dtype),
            nn.Tanh(),
            nn.Linear(hidden, hidden, device=device, dtype=dtype),
            nn.Tanh(),
            nn.Linear(hidden, self.OUTPUT_DIM, device=device, dtype=dtype),
        )

        self._initialize_weights()

    # ------------------------------------------------------------------------------
    # 2️⃣ Public API
    # ------------------------------------------------------------------------------
    def forward(self, t: Tensor) -> Tensor:  # noqa: D401, N802 (❓)
        """Forward pass.

        Parameters
        ----------
        t : torch.Tensor
            Input tensor of shape ``(..., 1)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(..., `)``.
        """
        return self.net(t)

    # ------------------------------------------------------------------------------
    # 3️⃣ Private helpers
    # ------------------------------------------------------------------------------
    def _initialize_weights(self) -> None:
        """Initialize all linear layers with Kaiming normal initialization (❓)."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="tanh")
                if module.bias is not None:  # pragma: no branch
                    nn.init.zeros_(module.bias)


# ------------------------------------------------------------------------------
# 4️⃣ Example usage/ test case
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    set_global_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(hidden=128, device=device)
    dummy: Tensor = torch.randn(2, 1, device=device)

    out: Tensor = model(dummy)
    print(out)
