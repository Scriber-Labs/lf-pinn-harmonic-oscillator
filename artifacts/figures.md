# Figure Analysis

## Figure 1 - Training Curve
![Training Curve](training.png)
> Training curve exhibits rapid initial descent followed by periodic spikes. 

The early descent reflects the network's ability to quickly
learn qualitative features rather than exact parameterization. This is followed with alternating regions
of rapid decent and constraint violations. 

Spikes reflect transient conflicts between trajectory fitting and antisymmetric
structure in the learned generator.
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
![Phase Space Trajectory](phase_space.png) ![Vector Field](phase_space_quiver.png)
> The PINN learns a single phase-space trajectory (green) consistent with the Hamiltonian equations of motion. The analytic flow field (blue streamlines) and contant-energy orbit (blue dotted)are shown for reference and are not learned by the model.
 
**Analytic flow fields** $$\dot{x}=p, \quad \dot{p}=-\omega^2 x $$

**Constant-energy orbit** $$H(x,p)=\text{const}$$

**PINN-Trajectory** $\rightarrow$ learned dynamics.
- What the PINN does *not* do:
  - Spiral outward $\rightarrow$ no energy source.
  - Spiral inward $\rightarrow$ no energy sink/dissapation.
- What the PINN does do:
  - Approaches a closed orbit.
  - Initially, it crosses streamlines (❓🤔seems to be nearly perpendicular to stream at each crossing).
    - Corresponds to the generator adjusting its alignment with respect to the streamlines.
  - It eventually approaches a closed orbit and remains approximately tangent to streamlines.

- [ ] Do my best to understand this:
  - The trajectory visually encodes:
    - Bivector-generated rotation
    - Sympletctic intuition
    - Hamiltonian flow without formal machinery✔️