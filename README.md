# lf-pinn-harmonic-oscillator
A **low-fidelity physics-informed neural network (PINN)** demonstrating how physics can guide learning via variational pricniples. 
This repository simluates a **1D simple harmonic oscillator (SHO)** with an unknown frequency while softly enforcing the equation of motion and energy conservation.
The purpose of this repo is to serve as a foundational teaching phodule in physics-informed machine learning (PIML), emphasizing **interpretability** and **parsimony** oveer accuracy or performance.

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

  $$ \mathcal{L}_{phys} = \big<(\ddot{x} + \omega^2 x)^2 \big> $$

## 🪏 Artifacts
- `trajectory.png`: learned $x(t)$
- `energy.png`: $H(t)$
- `notes.md`: reflection notes

## 5 Machine Learning Stages (Brunton-prescription) 
1. Problem formulation ✅
    > **Problem:** Can a neural function approximator recover physically meaningful motion via minimization of a variational resiudal, as opposed to fitting data?
2. Data collection and curation 
    - ❎ This repo is intentionally minimal and does not use any observational data to train the model.
    - Instead of observational input data, we use collocation points in time (gives synthetic, structure-driven input "data").
        - In this setup, the synthetic data we create is used to 'bake' our governing equation of interest (1D SHO) into the machine learning framework.
        - Note that the input "data" will be evaluated on a low-dimensional domain. 

    > **TL;DR:** This stage is deliberatily left degenerate. Instead of observational trajectoeries/data, the training data consists solely of simulated collocation points.
    
3. Neural architecture
    - ⚠️ implemented, but still need personally to bridge the conceptual justications. 
        - For now, we'll just say the current archtecture was inspired by Steve Brunton-style approaches to ML architecture design. 
    - Importantly, this model is not trying to learn physics from architectures. Rather, it is promoting physics via the loss funciton.
    - Overview of architecture choices:
        - Low-depth MLP
        - Scalar input $\rightarrow$ scalar output
        - Smooth activations (`tanh`)
        - No convolutions
        - No recurrence
        - No unneccessary/unjuistified inductive bias

4. Loss function ✅
    - The loss function $$\mathcal{L}_{phys} = \Big< (\ddot{x} + \omega^2x)^2\Big>$$ encodes:
        - Euler-Lagrange structure
        - Second-order dynamics
        - physical consistency
    - Rule of thumbe (Steve Brunton-inspired): 
        > Loss functions should refelct governing equations, not arbitrary error metrics.

5. Constrain optimization
    - ❎⚠️ like the neural architecture step, I'm still learning a lot of coneptual basics, and like the data curation step, this is intentionally left minimalistic and uses 'go-to' optimization routine:
        - Adam
        - Fixed learning rate
        - No scheduler (❓)

   - Additionally, we are using the *optimization should be revealing structure* (Steve-Brunton inspired) guideline as justification for blindly following 'standard' routines for our low-fidelity SHO PINN. Future repos will aim to craft unique implementations of this and other 'incomplete' steps. (Stay tuned!)

## Next Steps
- Create modesl that...
    - learns $\omega$ and treats it as a trainable parameter
    - or condition's the network on $\omega$ (❓)

## Final Thought
> *"The harmonic oscillatork is to physiucs what linear regression is to machine learning."*
