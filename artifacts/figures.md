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

## Figure 2 - Position Trajectories
![Trajectories](trajectory.png)
> Position component of the predicted trajectory $q_\theta(t)$ (green curve)
> and the ground truth trajectory $q(t)$ (blue dashed curve). 

Note that we use canonical coordinates $q(t)=x(t)$ and $p(t)=\dot{x}$, corresponding to position and momentum of a one-dimensional harmonic oscillator. 

---
## Figure 3 - Hamiltonian Evolution
![Hamiltonian Evolution](energy.png)
> The learned generator $G_\theta$ gives a vector field increasingly tangent to its own contours. From this emerges conservation enforced by antisymmetry rather than numeric accuracy.
> 
Three regions:
1. Sharp initial drop: Model discovers approximate invariant.
2. Small 'bump': Rebalancing between trajectory and structure losses.
3. Approximately flat region: Near-stationary energy error.
---
## Figure 4 - Phase Space Trajectory: Flow vs. Vectors
| Hamiltonian Streamlines | Hamiltonian Vector Field |
|:---:|:---:|
| ![Phase Space Trajectory](phase_space.png) | ![Vector Field](phase_space_quiver.png) |

> **Left:** The analytic flow (blue streamlines) shows the global topology of the conservation law. **Right:** The vector field (quiver plot) shows the local direction and magnitude of the Hamiltonian gradient.

**Analytic flow fields** $$\dot{x}=p, \quad \dot{p}=-\omega^2 x $$

**PINN-Trajectory** $\rightarrow$ learned dynamics.
- **Variable Convention:** In dynamical systems and PINNs, the predicted state is typically denoted as $\mathbf{z}_\theta(t) = [q_\theta(t), p_\theta(t)]^T$. Using $\mathbf{z}_\theta$ (bold for vector) is considered best practice.
