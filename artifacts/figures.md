# Figure Analysis

## Figure 1 - Training Curve
![Training Curve](training.png)
Two regions:
1. Fast early descent.
    - Reflects the networks ability to quickly learn qualitative features rather than exact parameterization.
    - Manifests due to a loss function dominated by global consistency constraints (ODE residuals, conservation structure). 
       - [ ] Need to understand this better.
2. Oscillatory spikes.
    - Demonstrates geometric reorganization whenever the network (briefly) violates symplectic consistency.
    - Arises due to competition between trajectory accuracy and structural consistency.
---

## Figure 2 - Trajectories (Predicted and Ground Truth)
![Trajectories](trajectory.png)
- [ ] Make sure you got the lengend correct.

---
## Figure 3 - Hamiltonian Evolution
![Hamiltonian Evolution](energy.png)
Three regions:
1. Sharp initial drop.
2. Small 'bump'.
3. Near-constant flat line.
---
## Figure 4 - Phase Space Trajectory Plotted over Hamiltonian Flow Fields 
![Phase Space Trajectory](phase_space.png)
> The PINN learns a single phase-space trajectory consistent with the Hamiltonian equations of motion. The analytic flow field (blue streamlines)  and contant-energy orbit (purple dotted)are shown for reference and are not learned by the model.
 
**Analytic flow fields** $$\dot{x}=p, \quad \dot{p}=-\omega^2 x $$

**Constant-energy orbit** $$H(x,p)=\text{const}$$

**PINN-Trajectory** $\rightarrow$ learned dynamics.

- [ ] Do my best to understand this:
  - The trajectory visually encodes:
    - Bivector-generated rotation
    - Sympletctic intuition
    - Hamiltonian flow without formal machiner✔️