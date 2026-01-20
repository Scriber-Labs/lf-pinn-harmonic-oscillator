# LF PINN Harmonic Oscillator Notes - Geometry Notes
> **Scriber Labs -**  Intuition-first notes connecting Hamiltionian mechanics, symplectic geometry, and low-fidelity PINNs. 

## Scope and Goals
These notes collect the geometric ideas that motivate the **low-fidelity PINN (LF-PINN)** for the 1D simple harmonic oscillator (SHO). 
The emphasis is on *structure* (symplectic area, bivectors, and generators of flow), not on building an exact symplectic integrator.  

## Terminology & Open Threads
- [ ] Vectors vs. covectors (and why momenta live naturally as covectors)
- [ ] Covectors in geometric/ Clifford algebra
- [ ] Hamiltonian mechanics from geometry (kinetic + potential $\rightarrow$ flow)
- [ ] Laplacian from Hamiltonian structure
- [ ] Duals of covectors (contrast with cross vs. wedge products)
- [ ] Show that $$J = \begin{bmatrix} 0 & 1 \\ -1 & 0 \end{bmatrix}$$ is the matrix representation of $dq\wedge dp$.
- [ ] Concept check: Is the phase-space pseudoscalar $I=dq\wedge dp$ a *differential-sized* version of $\mathbb{e}_1 \wedge \mathbb{e}_2$?
- [ ] What is a bivector contraction?
---

## Bilinear skew-symmetric map
A function $\omega : V\times V \rightarrow \mathbb{R}$ is **bilinear** and **skew-symmetric** if it is linear in
    $$ \forall u,v \in V, \quad \omega(u,v)=-\omega(v,u).$$

It is **bilinear** if it is linear in *each* argument.

> 🏡 **Take-Home Message:** Skew symmetry alone does not imply a function is bilinear. 

## Non-degeneracy
Let $V$ be a vector space over the real numbers, and let $\omega: V\times V \rightarrow \mathbb{R}$ be a bilinear function. 
The function $\omega$ is said to be **non-degenerate** iff
        
$$ \{ v \in V | \forall w\in V, \omega(v,w)=0\} = \{ 0 \} .$$

Equivalently, $\omega$ is a non-degenerate function if and only if the zero vector is the only vector that pairs to zero along with every other vector in $v\in V$.

> 💎 **Gem Take-away:** "Non-degeneracy means there exists an invertible mapping $\omega$ between vectors and covectors.

---

### Symplectic form
- Casual Definition: A **symplectic form** measures *signed area* in phase space $(\bf{q}, \bf{p})$ 
- Technical definition: A **symplectic form** is a smooth, closed, non-degenerate, 2-form that fixes the orientation and geometry of phase spce. 

For a 1D system in phase-space $(q,p)$, the corresponding symplectic form is $$ \omega = dq\wedge dp $$

Properties:
1. **Closed:** $d\omega = 0$ 
2. **Non-degenerate**
3. **Smooth** differential form 
   - [ ] Is this the same thing as 'well-behaved' functions in intro quantum mechanics courses?
4. **Antisymmetric**: $dq\wedge dp = -dp \wedge dq$

> 🏡 The antisymmetry property of phase space in physics is the root of:
 >  - Poisson brackets
 >  - Hamilton's equations
 >  - Conservation of phase-space volume

## Symplectic Forms and Harmonic Oscillator PINNs
<figure>
  <img src="assets/images/action_area.png" alt="Action as the area inside the energy contour." width="500">
  <figcaption><em>Figure 1:</em> Phase space trajectory of the harmonic oscillator.</figcaption>
</figure>

For the 1D harmonic oscillator PINN, the symplectic form $\omega = dq \wedge dp$ induces Hamilton's equations:
$$ \dot{q}= \frac{\partial H}{\partial p}, \quad \dot{p}=-\frac{\partial H}{\partial q}$$

where 

$$H(q,p) = \frac{p^2}{2m} + \frac{1}{2}k q^2 $$

is the Hamiltonian for the 1D harmonic oscillator. This is the actual constraint you want your PINN to respect. 


In this repository, "low-fidelity PINN" means that physical structure is encouraged via soft penalty terms in the loss function (rather than enforced exactly). The main point is to produce interpretible results; NOT to build a fully syumplectic 'neural' (❓) integrator. In other words, the models is not guaranteed to be symjplectic, however it is biased to wards the Hamiltonian.

According to chat-GPT, low-fidelity means (🤔 is this a technical description, or just approximate)...
- Penalizing violations of symplectic structure 
- It is not exactly enforced (🤔 what exactly (pardon the pun) does this mean?).

Examples of valid LF constraints:
- Penalizing the residuals...
    $$ \begin{align*}
        \partial_tq - \frac{\partial H}{\partial p},  \\ \\
        \partial_tp + \frac{\partial H}{\partial q}
        \end{align*}    $$

- Penalizing phase-space volume drift.

## Clifford Algebra Detour
In Clifford algebra $\omega = \partial_q \wedge \partial_p$ is just a bivector:
    $$ \partial_q \wedge \partial_p = \frac{1}{2}(\partial_q\partial_p - \partial_p\partial_q)$$

💎 Therefore, we can say $\omega$ generates rotations in phase space and Hamiltonian flow is a *bivector-generated motion*. 

### Bivector Operator as a Symplectic Matrix
The matrix $$J = \begin{bmatrix} 0 & 1 \\ -1 & 0 \end{bmatrix}$$ is the matrix representation of $dq\wedge dp$.
In Clifford algebra language:
- $J$ is equivalent to multiplication by the phase-space pseudoscalar.
- Poisson brackets are bivector contractions.
- Hamilton's equations are rotations generated by the bivector $\omega$. 

## 🏡 Take Home Messages


- The action $S$ is the area inside energy flows in phase space.