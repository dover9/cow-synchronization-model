# 🐄 Cow Synchronization: Modeling Reference Sheet

## 1. State Variables and Observable Modes

Each cow’s biological status is represented by:

$$
w = (x, y; \theta) \in [0,1] \times [0,1] \times \Theta
$$

where:
- $x \in [0,1]$ is the hidden **desire to eat**,
- $y \in [0,1]$ is the hidden **desire to lie down**,
- $\theta \in \Theta = \{ \mathcal{E}, \mathcal{R}, \mathcal{S} \}$ is the **observable behavior**:
  - $\mathcal{E}$ = Eating,
  - $\mathcal{R}$ = Ruminating (lying down),
  - $\mathcal{S}$ = Standing.

### Description
> Each cow is modeled with two hidden internal states — a desire to eat and a desire to lie down — both evolving over time. The observable behavior ($\theta$) depends on these internal needs and switches according to threshold rules. The full cow state $w$ includes both the hidden dynamics and the visible action.

## 2. Uncoupled Dynamics of a Single Cow

The internal state variables $(x, y)$ evolve differently depending on the cow’s observable behavior $\theta$.

**($\mathcal{E}$) Eating State:**

$$
\dot{x} = -\alpha_2 x, \quad \dot{y} = \beta_1 y
$$

**($\mathcal{R}$) Resting (Ruminating) State:**

$$
\dot{x} = \alpha_1 x, \quad \dot{y} = -\beta_2 y
$$

**($\mathcal{S}$) Standing State:**

$$
\dot{x} = \alpha_1 x, \quad \dot{y} = \beta_1 y
$$

### Description

> The dynamics of the cow’s hidden hunger ($x$) and lying desire ($y$) depend on its current observable behavior ($\theta$).  
> Parameters $\alpha_1, \alpha_2, \beta_1, \beta_2$ are all positive real numbers and represent biological rates:
> - $\alpha_1$: rate of increase of hunger,
> - $\alpha_2$: decay rate of hunger,
> - $\beta_1$: rate of increase of desire to lie down,
> - $\beta_2$: decay rate of desire to lie down.

---

## 3. Observable Behavior Switching Rules

The cow's observable behavior $\theta$ evolves according to the following rules based on the internal states $(x, y)$:

**Switch to Eating ($\mathcal{E}$)** if:

$$
\theta \in {\mathcal{R}, \mathcal{S}} \quad \text{and} \quad x = 1
$$

**Switch to Ruminating ($\mathcal{R}$)** if:

$$
\theta \in \{ \mathcal{E}, \mathcal{S} \} \quad \text{and} \quad x < 1, \quad y = 1
$$

**Switch to Standing ($\mathcal{S}$)** if:

$$
\theta \in \{ \mathcal{E}, \mathcal{R} \} \quad \text{and} \quad \left( y = \delta \, \text{and} \, x < 1 \quad \text{or} \quad x = \delta \, \text{and} \, y < 1 \right)
$$

where:
- $\delta$ is a small positive threshold (e.g., $\delta = 0.01$) used to prevent the cow from becoming stuck at the $(x, y) = (0, 0)$ point.

### Description

> Cows switch behaviors based on their internal hunger ($x$) and lying desire ($y$) levels.  
> - When hunger reaches its maximum, cows switch to eating.  
> - When lying desire reaches its maximum (and hunger is under control), cows switch to ruminating.  
> - When either need falls very low, cows switch to standing.

## 4. Discrete Dynamics and the Poincaré Map

Rather than continuously tracking cows at every moment, we focus on key events when their internal states $(x, y)$ reach important thresholds.

We define a special set $\Sigma$ where transitions occur:
- $\Sigma$ consists of points where hunger $x = 1$ or lying desire $y = 1$.
- These correspond to natural switching events (start eating, start ruminating).

We extend $\Sigma$ to $\Sigma'$ by also including:
- Points where hunger $x = \delta$ (low hunger threshold),
- Points where lying desire $y = \delta$ (low lying desire threshold).

Thus, $\Sigma'$ captures all important boundary crossings:
- Hunger maxed or low,
- Lying desire maxed or low.

---

The **discrete map** $g$:
- Takes a cow's current $(x,y,\theta)$ at the boundary $\Sigma'$,
- Predicts where the cow will land next,
- Determines the cow's new behavior.

Different cases (a)-(h) describe the possibilities depending on which threshold was crossed first:
- Whether $x$ or $y$ triggered the event,
- Whether the cow switches to eating, ruminating, or standing.

By focusing on transitions at $\Sigma'$, we simplify the analysis of synchronization and cow behavior patterns.

## 5. Discrete Mapping Rules (Cases a–h)

The discrete map $g$ determines how the cow's state $(x,y,\theta)$ evolves after hitting a boundary in $\Sigma'$.

Each case corresponds to hitting a boundary (either $x=1$, $y=1$, $x=\delta$, or $y=\delta$) and switching behavior accordingly.


**Case (a)**  
Starting from $\theta = \mathcal{E}$, hitting $x=1$.

If $y \geq \dfrac{\beta_1}{\alpha_2}$:

$$
g(x=1, \delta \leq y \leq 1; \mathcal{E}) = \left( \dfrac{\alpha_2}{\beta_1} y, \, 1; \, \mathcal{R} \right)
$$


**Case (b)**  
Starting from $\theta = \mathcal{E}$, hitting $x=1$.

If $y < \dfrac{\beta_1}{\alpha_2}$:

$$
g(x=1, \delta \leq y \leq 1; \mathcal{E}) = \left( \delta, \, \delta^{-\frac{\beta_1}{\alpha_2}} y; \, \mathcal{S} \right)
$$


**Case (c)**  
Starting from $\theta = \mathcal{R}$, hitting $y=1$.

If $x \geq \dfrac{\alpha_1}{\beta_2}$:

$$
g(\delta \leq x < 1, y=1; \mathcal{R}) = \left( 1, \, \dfrac{\beta_2}{\alpha_1} x; \, \mathcal{E} \right)
$$


**Case (d)**  
Starting from $\theta = \mathcal{R}$, hitting $y=1$.

If $x < \dfrac{\alpha_1}{\beta_2}$:

$$
g(\delta \leq x < 1, y=1; \mathcal{R}) = \left( \delta^{-\frac{\alpha_1}{\beta_2}} x, \, \delta; \, \mathcal{S} \right)
$$


**Case (e)**  
Starting from $\theta = \mathcal{S}$, hitting $x=\delta$.

If $y \leq \dfrac{\beta_1}{\alpha_1}$:

$$
g(x=\delta, \delta \leq y < 1; \mathcal{S}) = \left( 1, \, \delta^{-\frac{\beta_1}{\alpha_1}} y; \, \mathcal{E} \right)
$$


**Case (f)**  
Starting from $\theta = \mathcal{S}$, hitting $x=\delta$.

If $y > \dfrac{\beta_1}{\alpha_1}$:

$$
g(x=\delta, \delta \leq y < 1; \mathcal{S}) = \left( \dfrac{\alpha_1}{\beta_1} y, \, 1; \, \mathcal{R} \right)
$$


**Case (g)**  
Starting from $\theta = \mathcal{S}$, hitting $y=\delta$.

If $x \geq \dfrac{\alpha_1}{\beta_1}$:

$$
g(\delta < x < 1, y=\delta; \mathcal{S}) = \left( 1, \, \delta^{-\frac{\alpha_1}{\beta_1}} x; \, \mathcal{E} \right)
$$


**Case (h)**  
Starting from $\theta = \mathcal{S}$, hitting $y=\delta$.

If $x < \dfrac{\alpha_1}{\beta_1}$:

$$
g(\delta < x < 1, y=\delta; \mathcal{S}) = \left( \delta^{-\frac{\alpha_1}{\beta_1}} x, \, 1; \, \mathcal{R} \right)
$$

### Summary of Discrete Mapping Rules (Cases a–h)

The discrete mapping rules (cases a–h) govern how a cow’s internal state $(x, y)$ and observable behavior $\theta$ update when crossing a boundary in $\Sigma'$. 

Each rule describes a possible transition event:
- Based on which variable ($x$ or $y$) reached a critical threshold first,
- And depending on the cow’s current behavior ($\theta$).

The discrete map $g$ specifies:
- The cow’s new observable behavior (Eating, Ruminating, or Standing),
- The cow’s new internal state variables $(x, y)$ immediately after the transition.

Together, these rules allow modeling the cow’s full evolution as a sequence of continuous flows interrupted by discrete jumps.

### Note on Single-Cow Behavior

Analysis of the single-cow model shows that, depending on parameter choices, cows may converge to simple periodic orbits involving sequences of eating, ruminating, and standing. Stability properties and the long-term behavior of individual cows depend on the relative values of $\alpha_1$, $\alpha_2$, $\beta_1$, and $\beta_2$.

## 6. Coupled Dynamics for Synchronization

To model interactions between cows, we introduce a coupling mechanism based on the idea that cows become hungrier when they see others eating and have a greater desire to lie down when they see others lying down.

We define indicator functions on the set $\Theta = \{ \mathcal{E}, \mathcal{R}, \mathcal{S} \}$:

$$
\chi_{\psi}(\theta) =
\begin{cases}
1, & \text{if } \theta = \psi, \\\\
0, & \text{otherwise}.
\end{cases}
$$

Using these indicators, the single-cow dynamics can be rewritten compactly as:

$$
\dot{x} = \alpha(\theta) x, \quad \dot{y} = \beta(\theta) y,
$$

where

$$
\alpha(\theta) = -\alpha_2 \chi_{\mathcal{E}}(\theta) + \alpha_1 \chi_{\mathcal{R}}(\theta) + \alpha_1 \chi_{\mathcal{S}}(\theta),
$$

$$
\beta(\theta) = \beta_1 \chi_{\mathcal{E}}(\theta) - \beta_2 \chi_{\mathcal{R}}(\theta) + \beta_1 \chi_{\mathcal{S}}(\theta).
$$

---

To model interactions, we define the **eating influence** $e_i$ and **ruminating influence** $r_i$ for each cow $i$:

$$
e_i = \frac{1}{k_i} \sum_{j=1}^n a_{ij} \chi_{\mathcal{E}}(\theta_j)
$$

$$
r_i = \frac{1}{k_i} \sum_{j=1}^n a_{ij} \chi_{\mathcal{R}}(\theta_j)
$$

where:
- $a_{ij}(t) = 1$ if cow $i$ perceives cow $j$ at time $t$, and $0$ otherwise,
- $k_i = \sum_{j=1}^n a_{ij}$ is the number of cows visible to cow $i$.

---

Now, the **coupled dynamics** for each cow are:

$$
\dot{x}_i = \left( \alpha^{(i)}(\theta_i) + \sigma_x e_i \right) x_i
$$

$$
\dot{y}_i = \left( \beta^{(i)}(\theta_i) + \sigma_y r_i \right) y_i
$$

where:
- $a_{ij}(t) = 1$ if cow $i$ perceives cow $j$ at time $t$, and $0$ otherwise,
- $k_i = \sum\limits_{j=1}^{n} a_{ij}$ is the number of cows visible to cow $i$,
- $\sigma_x$ and $\sigma_y$ are non-negative coupling strengths.

<!-- 
Full LaTeX version for Overleaf (does not render on GitHub):

Now, for a herd of $n$ cows, indexed by $i$, the coupled dynamics are:

$$
\dot{x}_i = \left( \alpha^{(i)}(\theta_i) + \frac{\sigma_x}{k_i} \sum_{j=1}^{n} a_{ij} \chi_{\mathcal{E}}(\theta_j) \right) x_i
$$

$$
\dot{y}_i = \left( \beta^{(i)}(\theta_i) + \frac{\sigma_y}{k_i} \sum_{j=1}^{n} a_{ij} \chi_{\mathcal{R}}(\theta_j) \right) y_i
$$
-->

---

### Description

> Each cow evolves according to its own internal dynamics, modified by interactions with neighboring cows.  
> Cows feel hungrier when they observe others eating and feel more desire to lie down when they observe others ruminating.  
> The adjacency matrix $A$ defines the social network between cows at any given time.

## 7. Measuring Synchronization

To quantify the level of synchronization between cows, we track the times at which each cow switches observable behaviors.

For each cow $i$:
- Let $\tau^{(i)}$ be the sequence of times at which cow $i$ switches to the eating state $\mathcal{E}$.
- Let $\kappa^{(i)}$ be the sequence of times at which cow $i$ switches to the ruminating state $\mathcal{R}$.

---

### Pairwise Synchronization Measures

Given two cows $i$ and $j$, and assuming $\tau^{(i)}$ and $\tau^{(j)}$ are vectors of the same length $K$,  
the **eating synchronization error** between cows $i$ and $j$ is defined as:

$$
\Delta_{ij}^{\mathcal{E}} \equiv \left\langle \left| \tau_k^{(i)} - \tau_k^{(j)} \right| \right\rangle = \frac{1}{K} \sum_{k=1}^K \left| \tau_k^{(i)} - \tau_k^{(j)} \right|,
$$

where $\langle \cdot \rangle$ denotes time-averaging.

Similarly, the **ruminating synchronization error** is defined as:

$$
\Delta_{ij}^{\mathcal{R}} \equiv \left\langle \left| \kappa_k^{(i)} - \kappa_k^{(j)} \right| \right\rangle.
$$

Smaller values of $\Delta_{ij}^{\mathcal{E}}$ and $\Delta_{ij}^{\mathcal{R}}$ indicate stronger synchronization between cows.

---

### 📚 Group Synchronization Measures

For a herd of $n$ cows, the overall group synchronization is obtained by averaging over all cow pairs:

$$
\Delta^{\mathcal{E}} \equiv \frac{1}{n^2} \sum_{i,j} \Delta_{ij}^{\mathcal{E}},
$$

$$
\Delta^{\mathcal{R}} \equiv \frac{1}{n^2} \sum_{i,j} \Delta_{ij}^{\mathcal{R}}.
$$

The **aggregate synchronization measure** is then defined as:

$$
\Delta \equiv \Delta^{\mathcal{E}} + \Delta^{\mathcal{R}}.
$$

---

### Description

> Synchronization is measured by comparing the switching times between cows.  
> Lower synchronization error values imply that cows switch between behaviors (eating and ruminating) at more similar times.  
> The aggregate synchronization $\Delta$ captures the total mismatch across the herd.

## 8. Numerical Exploration of Herd Synchrony

We perform numerical simulations to investigate synchronization behavior in small herds.

### Simulation Setup for Two Coupled Cows

We consider a herd consisting of two cows with nearly identical but slightly mismatched parameters:

$$
\alpha_1^{(1,2)} = 0.05 \pm \epsilon, \quad \alpha_2^{(1,2)} = 0.1 \pm \epsilon,
$$

$$
\beta_1^{(1,2)} = 0.05 \pm \epsilon, \quad \beta_2^{(1,2)} = 0.125 \pm \epsilon,
$$

where $\epsilon$ is a small mismatch parameter.

We set:

$$
\delta = 0.25
$$

for the low-threshold switching value.

The coupling strengths $\sigma_x$ and $\sigma_y$ are varied to explore their effect on the degree of synchronization.

---

### Description

> Simulations examine how varying the mismatch $\epsilon$ and coupling strengths $\sigma_x, \sigma_y$ affects synchronization in the herd.  
> We begin with a two-cow system and can later extend to larger herds if desired.
