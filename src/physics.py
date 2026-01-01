from __future__ import annotations

from typing import Final, Protocol

import torch
from torch import Tensor, autograd, nn

__all__: list[str] = ["ho_variational_loss"]    # functions imported via 'from physics import *'

# ------------------------------------------------------------------------------
# 0️⃣ Typing helpers
# ------------------------------------------------------------------------------


#import torch

#def ho_variational_loss(model, t, omega=1.0):
 ##  x = model(t)
    
   # dx = torch.autograd.grad(
     #   x, t, grad_output=torch.ones_like(x), create_graph=True
    #)[0]
    
    #dx = torch.autograd.grad(
     #   dx, t, grad_output=torch.ones_like(dx), create_graph=True
   # )[0]
    
    #residual = ddx + omega**2 * x 
    #return torch.mean(residual**2)
