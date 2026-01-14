# lf-pinn-harmonic-oscillator
A **low-fidelity physics-informed neural network (PINN)** demonstrating how physics can guide learning via variational pricniples. 
This repository simulates a **1D simple harmonic oscillator (SHO)** with an unknown frequency while softly enforcing the equation of motion and energy conservation.
The purpose of this repo is to serve as a foundational teaching module in physics-informed machine learning (PIML), emphasizing **interpretability** and **parsimony** oveer accuracy or performance.

---

## 🗂️ Repo structure

```
lf-pinn-harmonic-oscillator/
├── README.md
├── requirements.txt
├── pyproject.toml
├── src/
│   ├── model.py          # neural network ansatz
│   ├── physics.py        # SHO + variational loss
│   ├── train.py          # training loop and CLI
│   └── utils.py          # helper functions (e.g., seeding)
├── notebooks/
│   └── demo.ipynb        # visual + narrative
└── artifacts/
    ├── training.png                # training curve
    ├── trajectory.png              # model output 
    ├── energy.png                  # Hamiltonian evolution
    ├── phase_space.png             # phase space with learned Hamiltonian flow
    ├── phase_space_quiver.png      # phase space with learned Hamiltonian vector field
    ├── figures.md                  # figure analysis
    └── notes.md                    # conceptual notes and reflections
```

---

## 🔰 Implementation Overview
### Physical system: Simple Harmonic oscillator (SHO)

We model a 1-D SHO using the Lagrangian,

  $$ L(x,\dot{x}) = \frac{1}{2}\dot{x}^2 - \frac{1}{2}\omega^2 x^2 .$$
  
### Neural ansatz

  $$ x_\theta(t) = \text{MLP}(t) $$

### Variational loss (low fidelity) 

Rather than solving exactly, the **Euler-Lagrange residual** is penalized at collation points in time.

  $$ \mathcal{L}_{phys} = \bigg< \bigg( \frac{d}{dt}\frac{\partial L}{\partial \dot{x}} - \frac{\partial L}{\partial x} \bigg)^2 \bigg> $$

Which can be simplified to:

  $$ \mathcal{L}_{phys} = \big<(\ddot{x} + \omega^2 x)^2 \big> $$

> This encourages the network to respect physical dynamics. Note that the physical dynamics we want the model to respect are not directly enforced - hence "low-fidelity".
---

## 5 Machine Learning Stages (Brunton-inspired) 
1. **Problem formulation ✔️**
    Can a neural function approximator recover physically meaningful motion via minimization of a **variational resiudal**, rather than fitting observed data?
   
3. **Data collection and curation ✖️**
    - Intentionally minimal: no observational trajectories.
    - Collocation points in time serve as synthetic "data" to embed physics into training.
    
4. **Neural architecture ⚠️**
    - Low-depth MLP, scalar input $\rightarrow$ scalar output, `tanh` activations.
    - No convolutions, recurrence, or unnecessary inductive biases.
    - Physics enters through the **loss function**, not the architecture.

6. **Loss function ✅**
    $$\mathcal{L}_{phys} = \Big< (\ddot{x} + \omega^2x)^2\Big>$$
    - Encodes Euler-Lagrange structure, second-order dynamics, and physical consistency.


8. **Constrained optimization ❎⚠️**
    -  Standard Adam optimizer with fixed learning rate.
    -  Optimization is intented to **reveal physical structure**, rather than fully customize for performance. 
    -  Future work: explore trainable $\omega$, unique optimizers, and richer PINN designs.


---

## 🔰 Quick Start

```bash
# Install dependencies
pip install -e .

# Train a model with default settings
python -m train

# Launch demo notebook
jupyter lab notebooks/demo.ipynb
```

Optional CLI flags:
```bash
python -m train --hidden 128 --epochs 5000 --n-points 200 --omega 1.0 --seed 42 --device cpu
```

---

## Next Steps
- Train models with learnable frequency $\omega$.
- Condition the network explicitly on $\omega$.
- Explore richer low-fidelity physics constraints and Hamiltonian structure preservation (e.g., energy conservation loss term).
- Explore ways to introduce inductive biases (limiations)
---

## ⚠️ Limitations
- **Temporal Resolution:** The model's accuracy is strictly dependent on the density of the collocation points ($n_\text{points}$).
    - If $\omega$ increases or the time window is increased, undersampling can lead to aliasing thus preventing the network successfully capturing the underlying oscillation.
- **Extrapolation (00D):** As a global function approximator, the MLP acts as an interpolator. 
  - Consequently, performance degrades rapidly outside the training window $[0, 2\pi]$ unless periodic inductive biases are introduced.
- **Soft Constraints:** Physics is enforced via a penalty term in the lose function ("soft constratin").
   - Unlike symplectic integrators, this model does not strictly conserve the Hamiltonian, **⚠️⚠️Need to visualize this -> increase scale on energy plot⚠️⚠️**) 
- **Spectral Bias:** Neural networks naturally learn lower-frequency components first. High-frequency oscillators may require specialized architectures or adoptive sampling (❓What is adoptive sampling?).

---

## 📚 Citations
- **PINNs Foundational Paper:** Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707.
- **Data-Driven Modeling:** Brunton, S. L., & Kutz, J. N. (2022). *Data-Driven Science and Engineering: Machine Learning, Dynamical Systems, and Control*. Cambridge University Press.
- **Variational Mechanics:** Goldstein, H., Poole, C., & Safko, J. (2001). *Classical Mechanics*. Pearson.

---

## Conceptual Notes
Refer to `artifacts/notes.md` for:
- **Symplectic forms:** $dq \wedge dp$ and Hamiltonian flow
- **Bilinear/skew-symmetric mappings**
- **Clifford algebra:** bivectors generating phase-space rotations
- Interpretation of low-fidelity PINNs: penalizing deviations from physical constraints, not exact enforcement



> *"The harmonic oscillator is to physics what linear regression is to machine learning."*

## Author
Lauren Shriver | Scriber Labs © 2025-2026
