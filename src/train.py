
import torch
from model import MLP
from physics import ho_variational_loss 

t = torch.linspace(0, 2*torch.pi, 200).view(-1, 1)

model = MLP()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(3000):
    opt.zero_grad()
    loss = ho_variational_loss(model, t)
    loss.backward()
    opt.step()
    
    if step % 500 == 0:
        print(f"step {step}, loss {loss.item():.3e}")