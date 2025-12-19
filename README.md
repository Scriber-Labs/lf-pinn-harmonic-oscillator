# lf-pinn-harmonic-oscillator
A minimal interpretable PINN-inspired simulator that demonstrates how physics enters learning via variational principles (without chasing accuracy).

## 🗂️ Repo structure
```
lf-pinn-harmonic-oscillator/
├── README.md
├── requirements.txt
├── src/
│   ├── model.py          # neural ansatz
│   ├── physics.py        # HO + variational loss
│   ├── train.py          # training loop
│   └── utils.py
├── notebooks/
│   └── demo.ipynb        # visual + narrative
└── artifacts/
    ├── trajectory.png
    ├── energy.png
    └── notes.md
```

## 🔰 Implementation Overview
### Physical system: Harmonic oscillator
  $$ L(x,\dot{x}) = \frac{1}{2}\dot{x}^2 - \frac{1}{2}\omega^2 x^2 $$
  
### Neural ansatz
  $$ x_\theta(t) = \text{MLP}(t) $$


### Variational loss (low fidelity) 
Rather than solving exactly, we penalize violation of Euler-Lagrange:

  $$ \mathcal{L}_{phys} = \bigg< \bigg( \frac{d}{dt}\frac{\partial L}{\partial \dot{x}} - \frac{\partial L}{\partial x} \bigg)^2 \bigg> $$

Which can be simplified to:

  $$ \mathcal{L}_{phys} = \big<(x+\omega^2 x)^2 \big>$$

## 🪏 Artifacts
- `trajectory.png`: learned $x(t)$
- `energy.png`: $H(t)$
- `notes.md`: reflection notes
