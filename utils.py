import numpy as np
import matplotlib.pyplot as plt

# ============================
# Plotting Utilities
# ============================
def plot_single_cow(hidden_states, obs_states, state_change_idxs, case_label="generic"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'hspace': 0.4})

    ax1.plot(hidden_states[:, 0], hidden_states[:, 1], linewidth=2)
    ax1.scatter(hidden_states[0, 0], hidden_states[0, 1], color='red', label='Initial State', s=50)
    # for idx in state_change_idxs:
    #      ax1.annotate(
    #          f"{['E', 'R', 'S'][obs_states[idx]]} (s={idx})",
    #          (hidden_states[idx, 0], hidden_states[idx, 1]),
    #          textcoords="offset points",
    #          xytext=(5, 5),
    #          ha='center',
    #          fontsize=8,      # smaller font
    #          alpha=0.7)        # slightly transparent
         
    ax1.set_title("Hidden State Trajectory", fontsize=16)
    ax1.set_xlabel("x", fontsize=14)
    ax1.set_ylabel("y", fontsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.grid(True)

    ax2.plot(obs_states, linewidth=2)
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(["E", "R", "S"], fontsize=12)
    ax2.set_xlabel("Time Step", fontsize=14)
    ax2.set_ylabel("Observable State", fontsize=14)
    ax2.set_title("Observable State Evolution", fontsize=16)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def plot_observable_states(states_1, states_2, start=3000, title="Observable States Over Time"):
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


def plot_synchrony_vs_sigma(sigma_vals, mean_E, std_E, mean_R, std_R, title="Synchrony vs Coupling Strength"):
    plt.figure(figsize=(8, 5))

    # Plot Δ^E with solid blue line and error bars
    plt.errorbar(sigma_vals, mean_E, yerr=std_E, label=r'$\Delta^{\mathcal{E}}$', color='blue', linewidth=1)

    # Plot Δ^R with dashed red line and error bars
    plt.errorbar(sigma_vals, mean_R, yerr=std_R, label=r'$\Delta^{\mathcal{R}}$', color='red', linestyle='--', linewidth=1)

    plt.xlabel(r'$\sigma_{x,y}$', fontsize=14)
    plt.ylabel('Synchrony Error', fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

# ============================
# Synchrony Metrics
# ============================
def get_transition_times(obs_states, target_state, min_duration=3):
    """
    Returns time indices where the cow *enters* `target_state` and remains there
    for at least `min_duration` steps.
    """
    transition_times = []
    t = 1
    while t < len(obs_states):
        if obs_states[t] == target_state and obs_states[t - 1] != target_state:
            # Check if cow stays in target_state long enough
            duration = 1
            while t + duration < len(obs_states) and obs_states[t + duration] == target_state:
                duration += 1
            if duration >= min_duration:
                transition_times.append(t)
                t += duration  # skip forward
            else:
                t += 1  # too short to count, just move on
        else:
            t += 1
    return transition_times


def compute_pairwise_synchrony(times_i, times_j, T, max_shift=10):
    """
    Computes the minimum average absolute difference between two time series,
    allowing integer shifts in alignment and normalizing by the total time.

    Parameters:
        times_i, times_j: Lists of time steps (e.g., when cow i and j enter Eating)
        T: Total simulation time (e.g., T = n_steps * dt)
        max_shift: Maximum number of shifts to align the two series

    Returns:
        Normalized minimum average absolute difference (float)
    """
    min_len = min(len(times_i), len(times_j))
    if min_len == 0:
        return float('inf')

    best_delta = float('inf')

    for shift in range(-max_shift, max_shift + 1):
        if shift >= 0:
            shifted_j = times_j[shift:]
            aligned_len = min(len(times_i), len(shifted_j))
            cropped_i = times_i[:aligned_len]
            cropped_j = shifted_j[:aligned_len]
        else:
            shifted_i = times_i[-shift:]
            aligned_len = min(len(shifted_i), len(times_j))
            cropped_i = shifted_i[:aligned_len]
            cropped_j = times_j[:aligned_len]

        if len(cropped_i) == 0:
            continue

        delta = np.mean(np.abs(np.array(cropped_i) - np.array(cropped_j))) / T
        best_delta = min(best_delta, delta)

    return best_delta


def compute_herd_synchrony(state_sequences, T, max_shift=10):
    """
    Computes average herd synchrony over all unordered cow pairs (i, j), i < j.

    Parameters:
        state_sequences: list of observable state arrays (1 per cow)
        T: total simulation time (e.g., steps x dt)
        max_shift: maximum time shift allowed when aligning transitions

    Returns:
        (avg_delta_E, avg_delta_R, total_delta)
    """
    n = len(state_sequences)
    tau_list = [get_transition_times(states, 0) for states in state_sequences]
    kappa_list = [get_transition_times(states, 1) for states in state_sequences]

    total_delta_E = 0
    total_delta_R = 0
    count = 0

    for i in range(n):
        for j in range(i + 1, n):  # only unordered pairs (i < j)
            delta_E = compute_pairwise_synchrony(tau_list[i], tau_list[j], T, max_shift)
            delta_R = compute_pairwise_synchrony(kappa_list[i], kappa_list[j], T, max_shift)
            total_delta_E += delta_E
            total_delta_R += delta_R
            count += 1

    if count == 0:
        return float("inf"), float("inf"), float("inf")

    avg_delta_E = total_delta_E / count
    avg_delta_R = total_delta_R / count
    total = avg_delta_E + avg_delta_R

    return avg_delta_E, avg_delta_R, total

# ============================
# Adjacency Matrix Builders
# ============================
def build_grid_adjacency(rows, cols):
    """
    Builds an adjacency matrix for a 2D grid of cows with shape (rows x cols).
    Each cow is connected to its immediate neighbors (up, down, left, right).

    Example layout for rows=3, cols=4:

        0  —  1  -  2  -  3
        |     |     |     |
        4  -  5  -  6  -  7
        |     |     |     |
        8  -  9  - 10  - 11

    Returns:
        A: numpy array of shape (rows * cols, rows * cols)
    """
    n = rows * cols
    A = np.zeros((n, n), dtype=int)

    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j

            if i > 0:
                up = (i - 1) * cols + j
                A[idx, up] = 1
            if i < rows - 1:
                down = (i + 1) * cols + j
                A[idx, down] = 1
            if j > 0:
                left = i * cols + (j - 1)
                A[idx, left] = 1
            if j < cols - 1:
                right = i * cols + (j + 1)
                A[idx, right] = 1

    return A

def build_random_adjacency(n_cows, p=0.3, seed=None):
    """
    Builds a random symmetric adjacency matrix for n cows.
    Each edge is included with probability p.

    Parameters:
        n_cows: int - number of cows
        p: float - probability of connection
        seed: int or None - random seed for reproducibility

    Returns:
        A: numpy array of shape (n_cows, n_cows)
    """
    if seed is not None:
        np.random.seed(seed)

    A = np.zeros((n_cows, n_cows), dtype=int)

    for i in range(n_cows):
        for j in range(i + 1, n_cows):
            if np.random.rand() < p:
                A[i, j] = 1
                A[j, i] = 1  # make it symmetric

    return A