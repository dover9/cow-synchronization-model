import numpy as np
import matplotlib.pyplot as plt
from cow import Cow
from cowherd import CowHerd

def simulate_one_cow(initial_conds: dict = {}, case_label: str = "generic"):
    """Simulate one cow for 1000 steps.
    Make two plots: one of hidden state over time, one of observable state over time.
    """
    cow = Cow(**initial_conds)
    steps = 5000 #15000
    hidden_states = np.zeros((steps, 2))
    obs_states = np.zeros(steps, dtype=int)
    state_change_idxs = [0]

    for i in range(steps):
        hidden_states[i] = cow.x, cow.y
        obs_states[i] = ["E", "R", "S"].index(cow.obs_state)
        # cow.update_hidden_state()
        cow.update_hidden_state(stepsize=0.1)
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


def simulate_two_cows(timesteps=10000, stepsize=.5, sigma_x=0.045, sigma_y=0.045):
    # Create two cows with nearly identical parameters
    epsilon = 0.001
    params1 = (0.05 + epsilon, 0.1 + epsilon, 0.05 + epsilon, 0.125 + epsilon)
    params2 = (0.05 - epsilon, 0.1 - epsilon, 0.05 - epsilon, 0.125 - epsilon)

    cow1 = Cow(params=params1, init_obsstate="E", delta=0.25)
    cow2 = Cow(params=params2, init_obsstate="E", delta=0.25)

    # Fully connected adjacency matrix
    adjacency = np.array([
        [0, 1],
        [1, 0]
    ])

    herd = CowHerd([cow1, cow2], adjacency, sigma_x=sigma_x, sigma_y=sigma_y)

    # Track observable states over time
    states_1 = []
    states_2 = []

    for _ in range(timesteps):
        states_1.append(["E", "R", "S"].index(cow1.obs_state))
        states_2.append(["E", "R", "S"].index(cow2.obs_state))
        herd.step(stepsize)

    return np.array(states_1), np.array(states_2)

def plot_observable_states(states_1, states_2, start=0, end=None):
    if end is None:
        end = len(states_1)

    time = range(start, end)
    plt.figure(figsize=(10, 4))
    plt.plot(time, states_1[start:end], label="Cow 1", alpha=0.8)
    plt.plot(time, states_2[start:end], label="Cow 2", alpha=0.8)
    plt.yticks([0, 1, 2], ["E", "R", "S"])
    plt.xlabel("Time step")
    plt.ylabel("Observable state")
    plt.title("Observable States Over Time for Two Uncoupled Cows")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_observable_states_stylized(states_1, states_2, start=3000, title="Observable States Over Time"):
    timesteps = np.arange(start, start + len(states_1))

    def get_state_change_points(states):
        return [0] + [i for i in range(1, len(states)) if states[i] != states[i - 1]]

    change_points_1 = get_state_change_points(states_1)
    change_points_2 = get_state_change_points(states_2)

    plt.figure(figsize=(10, 4))

    # Plot full lines
    plt.plot(timesteps, states_1, linestyle='-', color='red', linewidth=1, label='Cow 1')
    plt.plot(timesteps, states_2, linestyle='--', color='black', linewidth=1, label='Cow 2')

    # Plot markers only at state changes
    plt.plot(timesteps[change_points_1], np.array(states_1)[change_points_1], 'ro', markersize=6)
    plt.plot(timesteps[change_points_2], np.array(states_2)[change_points_2], 'kx', markersize=6)

    plt.yticks([0, 1, 2], ["E", "R", "S"])
    plt.xlabel("Time")
    plt.ylabel("Observable state")
    plt.title(title)
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # simulate_one_cow()
    # simulate_periodic_orbit_A()
    simulate_periodic_orbit_B()
    # simulate_periodic_orbit_C()
    # simulate_periodic_orbit_D()
    # full_states_1, full_states_2 = simulate_two_cows()

    # Select time slice
    start = 3000
    end = 3200
    states_1 = full_states_1[start:end]
    states_2 = full_states_2[start:end]

    # Plot using the stylized function
    # plot_observable_states_stylized(states_1, states_2, start=start, title="Observable States")