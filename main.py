import numpy as np
import matplotlib.pyplot as plt

class Cow:
    """One cow. At this point covers paper up to but not including 2.3"""

    def __init__(self,
                 init_hiddenstate: tuple[float, float] = None,
                 init_obsstate: str = None,
                 params: tuple[float, float, float, float] = None,
                 delta: float = 0.01):
        if init_hiddenstate is None:
            # self.x = np.random.uniform(0, 1)
            # self.y = np.random.uniform(0, 1)
            self.x = 1
            self.y = np.random.uniform(delta, 1)
        else:
            self.x, self.y = init_hiddenstate

        if init_obsstate is None:
            # self.obs_state = np.random.choice(["E", "R", "S"])
            self.obs_state = "E"
        else:
            self.obs_state = init_obsstate

        if params is None:
            self.alpha_1 = np.random.uniform(0, 0.1)
            self.alpha_2 = np.random.uniform(0, 0.1)
            self.beta_1 = np.random.uniform(0, 0.1)
            self.beta_2 = np.random.uniform(0, 0.1)
        else:
            self.alpha_1, self.alpha_2, self.beta_1, self.beta_2 = params

        self.delta = delta  # minimum hiddenstate values before switching

    def hidden_state_derivs(self) -> tuple[float, float]:
        """Return the derivatives of the hidden state."""
        # eqns 3-5
        if self.obs_state == "E":
            dx = -self.alpha_2 * self.x
            dy = self.beta_1 * self.y
        elif self.obs_state == "R":
            dx = self.alpha_1 * self.x
            dy = -self.beta_2 * self.y
        elif self.obs_state == "S":
            dx = self.alpha_1 * self.x
            dy = self.beta_1 * self.y
        return dx, dy
    
    def update_hidden_state(self, stepsize: float = 0.01) -> None:
        """Update hidden state by taking an Euler step."""
        # assert stepsize < self.delta  # ensure we don't take too big a step
        dx, dy = self.hidden_state_derivs()
        self.x += stepsize * dx
        self.y += stepsize * dy
        if self.x > 1:
            self.x = 1
        if self.y > 1:
            self.y = 1
            
    def next_obs_state(self) -> bool:
        """Update observable state. Returns True if the state has changed."""
        # eqn 6
        if self.obs_state in ("R", "S") and self.x >= 1:
            self.obs_state = "E"
            return True
        elif self.obs_state in ("E", "S") and self.x < 1 and self.y >= 1:
            self.obs_state = "R"
            return True
        elif (self.obs_state in ("E", "R") 
              and ((self.x < 1 and self.y <= self.delta) # the leq prevents oob
                   or (self.x <= self.delta and self.y < 1))):
            self.obs_state = "S"
            return True
        elif (self.x > 1 and self.y > 1 and self.obs_state != "S"):
            # tiebreaker. cf 3.0
            self.obs_state = "S"
            return True
        else:
            return False


def simulate_one_cow(initial_conds: dict = {}, case_label: str = "generic"):
    """Simulate one cow for 1000 steps.
    Make two plots: one of hidden state over time, one of observable state over time.
    """
    cow = Cow(**initial_conds)
    steps = 15000
    hidden_states = np.zeros((steps, 2))
    obs_states = np.zeros(steps, dtype=int)
    state_change_idxs = [0]

    for i in range(steps):
        hidden_states[i] = cow.x, cow.y
        obs_states[i] = ["E", "R", "S"].index(cow.obs_state)
        cow.update_hidden_state()
        if cow.next_obs_state():
            print(f"State changed to {cow.obs_state} at step {i}")
            state_change_idxs.append(i)

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'hspace': 0.4})

    # Hidden state trajectory
    ax1.plot(hidden_states[:, 0], hidden_states[:, 1], linewidth=2)
    ax1.scatter(hidden_states[0, 0], hidden_states[0, 1], color='red', label='Initial State', s=50)
    for idx in state_change_idxs:
        ax1.annotate(
            f"{['E', 'R', 'S'][obs_states[idx]]} (s={idx})",
            (hidden_states[idx, 0], hidden_states[idx, 1]),
            textcoords="offset points",
            xytext=(5, 5),
            ha='center',
            fontsize=8,      # smaller font
            alpha=0.7)        # slightly transparent

    ax1.set_title("Hidden State Trajectory", fontsize=16)
    ax1.set_xlabel("x", fontsize=14)
    ax1.set_ylabel("y", fontsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.grid(True)

    # Observable state over time
    ax2.plot(obs_states, linewidth=2)
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(["E", "R", "S"], fontsize=12)
    ax2.set_xlabel("Time Step", fontsize=14)
    ax2.set_ylabel("Observable State", fontsize=14)
    ax2.set_title("Observable State Evolution", fontsize=16)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.grid(True)

    plt.tight_layout()

    # Save plot
    save_path = f"figures/periodic_orbit_case_{case_label}.png"
    plt.savefig(save_path, dpi=300)
    print(f"[Info] Saved figure to {save_path}")

    plt.show()

    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))  # Increase figure size for more vertical space
    # ax1.plot(hidden_states[:, 0], hidden_states[:, 1])
    # ax1.scatter(hidden_states[0, 0], hidden_states[0, 1], color='red', label='Initial State')
    # for idx in state_change_idxs:
    #     ax1.annotate(f"{['E', 'R', 'S'][obs_states[idx]]} (s={idx})", (hidden_states[idx, 0], hidden_states[idx, 1]),
    #                  textcoords="offset points", xytext=(5, 5), ha='center')
    # ax1.set_title("Hidden state")
    # ax2.plot(obs_states)
    # ax2.set_yticks([0, 1, 2])
    # ax2.set_yticklabels(["E", "R", "S"])
    # ax2.set_title("Observable state")
    # plt.tight_layout()  # Adjust layout to increase spacing between plots
    # plt.show()


def simulate_periodic_orbit_A():
    """Simulate a periodic orbit by setting params to match eqn 12"""
    alpha_1 = np.random.uniform(0, 0.1)
    beta_1 = np.random.uniform(0, 0.1)
    alpha_2 = np.random.uniform(0, 0.1)
    beta_2 = alpha_1 * beta_1 / alpha_2
    simulate_one_cow(
        initial_conds={
            "params": (alpha_1, alpha_2, beta_1, beta_2)
        },
        case_label="A"
    )


def simulate_periodic_orbit_B():
    """Simulate Case B: periodic orbit E -> R -> S -> E.

    Notes:
        - Conditions:
            (alpha2 / alpha1) * (beta2 / beta1) > 1
            If beta1 < alpha2, then 1/alpha1 + 1/alpha2 >= 1/beta1 + 1/beta2
        - With these parameters, the orbit is stable but not asymptotically stable (alpha2 > alpha1).
    """
    alpha_1 = 0.03
    alpha_2 = 0.08
    beta_1 = 0.05
    beta_2 = 0.09

    # Condition checks
    prod_check = (alpha_2 / alpha_1) * (beta_2 / beta_1)
    assert prod_check > 1, f"Condition failed: prod_check = {prod_check}"
    if beta_1 < alpha_2:
        sum_alpha_inv = 1/alpha_1 + 1/alpha_2
        sum_beta_inv = 1/beta_1 + 1/beta_2
        assert sum_alpha_inv >= sum_beta_inv, f"Condition failed: sum_alpha_inv = {sum_alpha_inv}, sum_beta_inv = {sum_beta_inv}"

    simulate_one_cow(
        initial_conds={
            "params": (alpha_1, alpha_2, beta_1, beta_2)
        }, 
        case_label="B"
    )


def simulate_periodic_orbit_C():
    """Simulate Case C: periodic orbit E -> S -> R -> E.

    Notes:
        - Conditions:
            (alpha2 / alpha1) * (beta2 / beta1) > 1
            1/alpha1 + 1/alpha2 < 1/beta1 + 1/beta2
            beta1 < alpha2
        - For asymptotic stability, require beta2 < beta1.
    """
    # Hard-coded values satisfying stability conditions
    # Note: beta2 > beta1 → orbit is stable but NOT asymptotically stable
    alpha_1 = 0.04
    alpha_2 = 0.07
    beta_1 = 0.03
    beta_2 = 0.08                                   

    # Validate conditions
    prod_check = (alpha_2 / alpha_1) * (beta_2 / beta_1)
    sum_alpha_inv = 1 / alpha_1 + 1 / alpha_2
    sum_beta_inv = 1 / beta_1 + 1 / beta_2
    assert prod_check > 1, f"Condition failed: prod_check = {prod_check}"
    assert sum_alpha_inv < sum_beta_inv, f"Condition failed: sum_alpha_inv = {sum_alpha_inv}, sum_beta_inv = {sum_beta_inv}"
    assert beta_1 < alpha_2, f"Condition failed: beta_1 = {beta_1}, alpha_2 = {alpha_2}"

    # Special initial condition, always start at x = 1
    delta = 0.01                                    # Minimum hidden state value from Cow class
    y0 = delta ** (sum_alpha_inv / sum_beta_inv)
    x0 = 1                                        

    simulate_one_cow(
        initial_conds={
            "params": (alpha_1, alpha_2, beta_1, beta_2),
            "init_hiddenstate": (x0, y0),
            "init_obsstate": "E"
        },
        case_label="C"
    )


def simulate_periodic_orbit_D():
    """Simulate Case D: periodic orbit E -> S -> R -> S -> E.

    Notes:
        - Conditions:
            (alpha2 / alpha1) * (beta2 / beta1) > 1
            1/alpha1 + 1/alpha2 = 1/beta1 + 1/beta2
            beta1 < alpha2
        - x0 = 1
        - y0 must satisfy: delta < y0 < delta^(beta1/alpha2)
        - All orbits are stable but not asymptotically stable.
    """
    alpha_1 = 0.04
    alpha_2 = 0.08
    beta_1 = 0.05
    beta_2 = 1 / (1/alpha_1 + 1/alpha_2 - 1/beta_1)  # 0.05714 approx

    # Checks
    prod_check = (alpha_2 / alpha_1) * (beta_2 / beta_1)
    sum_alpha_inv = 1/alpha_1 + 1/alpha_2
    sum_beta_inv = 1/beta_1 + 1/beta_2
    assert prod_check > 1, f"Condition failed: prod_check = {prod_check}"
    assert np.isclose(sum_alpha_inv, sum_beta_inv, atol=1e-5), f"Condition failed: sums not equal: {sum_alpha_inv} vs {sum_beta_inv}"
    assert beta_1 < alpha_2, f"Condition failed: beta1 = {beta_1}, alpha2 = {alpha_2}"

    delta = 0.01
    lower_bound = delta
    upper_bound = delta ** (beta_1 / alpha_2)
    y0 = (lower_bound + upper_bound) / 2

    x0 = 1

    simulate_one_cow(
        initial_conds={
            "params": (alpha_1, alpha_2, beta_1, beta_2),
            "init_hiddenstate": (x0, y0),
            "init_obsstate": "E"
        },
        case_label="D"
    )

if __name__ == "__main__":
    # simulate_one_cow()
    simulate_periodic_orbit_A()
    simulate_periodic_orbit_B()
    simulate_periodic_orbit_C()
    simulate_periodic_orbit_D()
