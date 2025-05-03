import numpy as np
import matplotlib.pyplot as plt
from cow import Cow
from cowherd import CowHerd

def simulate_one_cow(initial_conds: dict = {}, steps: int = 15000, stepsize: float = 0.1):
    cow = Cow(**initial_conds)
    hidden_states = np.zeros((steps, 2))
    obs_states = np.zeros(steps, dtype=int)
    state_change_idxs = [0]

    for i in range(steps):
        hidden_states[i] = cow.x, cow.y
        obs_states[i] = ["E", "R", "S"].index(cow.obs_state)
        cow.update_hidden_state(stepsize=stepsize)
        if cow.next_obs_state():
            state_change_idxs.append(i)

    return hidden_states, obs_states, state_change_idxs


def simulate_periodic_orbit_A():
    """Simulate a periodic orbit by setting params to match eqn 12"""
    alpha_1 = np.random.uniform(0, 0.1)
    beta_1 = np.random.uniform(0, 0.1)
    alpha_2 = np.random.uniform(0, 0.1)
    beta_2 = alpha_1 * beta_1 / alpha_2
    
    return simulate_one_cow(initial_conds={"params": (alpha_1, alpha_2, beta_1, beta_2)})


def simulate_periodic_orbit_B():
    """Simulate Case B: periodic orbit E -> R -> S -> E.

    Notes:
        - Conditions:
            (alpha2 / alpha1) * (beta2 / beta1) > 1
            If beta1 < alpha2, then 1/alpha1 + 1/alpha2 >= 1/beta1 + 1/beta2
        - With these parameters, the orbit is stable but not asymptotically stable (alpha2 > alpha1).
    """
    alpha_1 = 0.08
    alpha_2 = 0.03
    beta_1 = 0.03
    beta_2 = 0.09
    delta = 0.01
    x0 = 1
    exponent = (1 + beta_1 / beta_2) / (1 + alpha_2 / alpha_1)
    y0 = delta ** exponent

    # Condition checks
    prod_check = (alpha_2 / alpha_1) * (beta_2 / beta_1)
    assert prod_check > 1, f"Condition failed: prod_check = {prod_check}"
    if beta_1 < alpha_2:
        sum_alpha_inv = 1 / alpha_1 + 1 / alpha_2
        sum_beta_inv = 1 / beta_1 + 1 / beta_2
        assert sum_alpha_inv >= sum_beta_inv, f"Condition failed: sum_alpha_inv = {sum_alpha_inv}, sum_beta_inv = {sum_beta_inv}"

    return simulate_one_cow(
        initial_conds={
            "params": (alpha_1, alpha_2, beta_1, beta_2),
            "init_hiddenstate": (x0, y0),
            "init_obsstate": "E"
        }
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

    return simulate_one_cow(
        initial_conds={
            "params": (alpha_1, alpha_2, beta_1, beta_2),
            "init_hiddenstate": (x0, y0),
            "init_obsstate": "E"
        }
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

    return simulate_one_cow(
        initial_conds={
            "params": (alpha_1, alpha_2, beta_1, beta_2),
            "init_hiddenstate": (x0, y0),
            "init_obsstate": "E"
        },
    )


def simulate_two_cows(epsilon=0.001, timesteps=10000, stepsize=.5, sigma_x=0.045, sigma_y=0.045):
    # Create two cows with nearly identical parameters
    params1 = (0.05 + epsilon, 0.1 + epsilon, 0.05 + epsilon, 0.125 + epsilon)
    params2 = (0.05 - epsilon, 0.1 - epsilon, 0.05 - epsilon, 0.125 - epsilon)
    
    cow1 = Cow(params=params1, init_obsstate="E", delta=0.25)
    cow2 = Cow(params=params2, init_obsstate="E", delta=0.25)
    
    # # Fixed initial internal state: x = 1, y = 0.1
    # init_state = (1.0, 0.1)
    # params1 = params2 = (0.05, 0.1, 0.05, 0.125)
    # cow1 = Cow(params=params1, init_obsstate="E", init_hiddenstate=init_state, delta=0.25)
    # cow2 = Cow(params=params2, init_obsstate="E", init_hiddenstate=init_state, delta=0.25)

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


def simulate_herd(n_cows=10, A=None, timesteps=10000, stepsize=0.5,
                  sigma_x=0.05, sigma_y=0.05,
                  param_base=(0.05, 0.1, 0.05, 0.125),
                  param_noise=0.001,
                  init_state=(1.0, 0.1),
                  delta=0.25):
    """
    Simulates a herd of n cows with slight parameter variation and coupling.

    Parameters:
        A: (n_cows x n_cows) adjacency matrix. If None, defaults to fully connected.

    Returns:
        List of observable state sequences, one per cow
    """
    base = np.array(param_base)
    cows = []

    for _ in range(n_cows):
        # Add slight noise to parameters
        noise = np.random.uniform(-param_noise, param_noise, size=4)
        params = tuple(base + noise)
        cow = Cow(params=params, init_obsstate="E", init_hiddenstate=init_state, delta=delta)
        cows.append(cow)

    # Fully connected adjacency matrix (no self-links)
    if A is None:
        A = np.ones((n_cows, n_cows)) - np.eye(n_cows)

    # Create Herd
    herd = CowHerd(cows, A, sigma_x=sigma_x, sigma_y=sigma_y)

    state_history = [[] for _ in range(n_cows)]

    for _ in range(timesteps):
        for i, cow in enumerate(cows):
            state_history[i].append(["E", "R", "S"].index(cow.obs_state))
        herd.step(stepsize)

    return state_history
