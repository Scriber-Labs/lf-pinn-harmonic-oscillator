# Figure Analysis

## Figure 1 - Training Curve
![Training Curve](training.png)
> Training loss evolves via repeated projection onto a Hamiltonian-consistent manifold.
>Produces periodic constraint-violation spikes followed by rapid geometric correction. 
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
- [ ] Is this the $q(t)$ 'component' of a trajectory on phase space?
- [ ] Is there any value gained in plotting the $p(t)$ component?

---
## Figure 3 - Hamiltonian Evolution
![Hamiltonian Evolution](energy.png)
> The learned generator $G_\theta$ gives a vector field increasingly tangent to its own contours. From this emerges conservation enforced by antisymmetry rather than numeric accuracy.
> 
Three regions:
1. Sharp initial drop.
2. Small 'bump'.
3. Near-constant flat line.
---
## Figure 4 - Phase Space Trajectory Plotted over Hamiltonian Flow Fields 
![Phase Space Trajectory](phase_space.png)
> The PINN learns a single phase-space trajectory (green) consistent with the Hamiltonian equations of motion. The analytic flow field (blue streamlines) and contant-energy orbit (blue dotted)are shown for reference and are not learned by the model.
 
**Analytic flow fields** $$\dot{x}=p, \quad \dot{p}=-\omega^2 x $$

**Constant-energy orbit** $$H(x,p)=\text{const}$$

**PINN-Trajectory** $\rightarrow$ learned dynamics.
- What the PINN does *not* do:
  - Spiral outward $\rightarrow$ no energy source.
  - Spiral inward $\rightarrow$ no energy sink/dissapation.
- What the PINN does do:
  - Approaches a closed orbit.
  - Initially, it crosses streamlines (seems to be nearly perpendicular to stream at each crossing).
    - Corresponds to the generator adjusting its alignment with respect to the streamlines.
  - It eventually approaches a closed orbit and remains approximately tangent to streamlines.

- [ ] Do my best to understand this:
  - The trajectory visually encodes:
    - Bivector-generated rotation
    - Sympletctic intuition
    - Hamiltonian flow without formal machinery✔️