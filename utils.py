import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

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

    plt.figure(figsize=(6, 3))

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

TOPOLOGY_LABELS = {
    'full': 'Fully Connected',
    'grid': '2D Grid',
    'random_p0.3': 'Random (p = 0.3)',
    'random_p0.75': 'Dense Random (p = 0.75)',
    'WALKING_COWHERD': 'Walking Cowherd',
}

def plot_synchrony_vs_sigma(sigma_vals, mean_E, std_E, mean_R, std_R, topology, n_cows):
    title = f"{n_cows}-Cow Herd — {TOPOLOGY_LABELS.get(topology, topology)} Topology"

    plt.figure(figsize=(8, 5))
    plt.plot(sigma_vals, mean_E, label=r'$\Delta^{\mathcal{E}}$ (Eating)', color='tab:blue')
    plt.fill_between(sigma_vals,
                     np.array(mean_E) - np.array(std_E),
                     np.array(mean_E) + np.array(std_E),
                     alpha=0.2, color='tab:blue')

    plt.plot(sigma_vals, mean_R, label=r'$\Delta^{\mathcal{R}}$ (Resting)', color='tab:red')
    plt.fill_between(sigma_vals,
                     np.array(mean_R) - np.array(std_R),
                     np.array(mean_R) + np.array(std_R),
                     alpha=0.2, color='tab:red')

    plt.xlabel("Coupling Strength ($\sigma$)")
    plt.ylabel("Mean Synchrony Error (minutes)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"synchrony_{topology}_{n_cows}cows.png")
    plt.close()


def plot_synchrony_vs_viz(vision_vals, mean_E, std_E, mean_R, std_R, topology, n_cows,
                           filename: str = None):
    title = f"{n_cows}-Cow Herd — {TOPOLOGY_LABELS.get(topology, topology)} Topology"

    plt.figure(figsize=(8, 5))
    plt.plot(vision_vals, mean_E, label=r'$\Delta^{\mathcal{E}}$ (Eating)', color='tab:blue')
    plt.fill_between(vision_vals,
                     np.array(mean_E) - np.array(std_E),
                     np.array(mean_E) + np.array(std_E),
                     alpha=0.2, color='tab:blue')

    plt.plot(vision_vals, mean_R, label=r'$\Delta^{\mathcal{R}}$ (Resting)', color='tab:red')
    plt.fill_between(vision_vals,
                     np.array(mean_R) - np.array(std_R),
                     np.array(mean_R) + np.array(std_R),
                     alpha=0.2, color='tab:red')

    plt.xlabel("Cow Vision Radius")
    plt.ylabel("Mean Synchrony Error (minutes)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if filename is None:
        filename = f"synchrony_{topology}_{n_cows}cows.png"
    plt.savefig(filename)
    plt.close()


def plot_synchrony_vs_mvmt(movement_vals, mean_E, std_E, mean_R, std_R, topology, n_cows,
                           filename: str = None):
    title = f"{n_cows}-Cow Herd — {TOPOLOGY_LABELS.get(topology, topology)} Topology"

    plt.figure(figsize=(8, 5))
    plt.plot(movement_vals, mean_E, label=r'$\Delta^{\mathcal{E}}$ (Eating)', color='tab:blue')
    plt.fill_between(movement_vals,
                     np.array(mean_E) - np.array(std_E),
                     np.array(mean_E) + np.array(std_E),
                     alpha=0.2, color='tab:blue')

    plt.plot(movement_vals, mean_R, label=r'$\Delta^{\mathcal{R}}$ (Resting)', color='tab:red')
    plt.fill_between(movement_vals,
                     np.array(mean_R) - np.array(std_R),
                     np.array(mean_R) + np.array(std_R),
                     alpha=0.2, color='tab:red')

    plt.xlabel("Cow Step Size")
    plt.ylabel("Mean Synchrony Error (minutes)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if filename is None:
        filename = f"synchrony_{topology}_{n_cows}cows.png"
    plt.savefig(filename)
    plt.close()


def plot_synchrony_vs_vizmvmt(vision_vals, movement_vals, mean_E, std_E, mean_R, std_R,
                                topology, n_cows, filename: str = None):
    # 3d plotly graph
    fig = go.Figure()

    fig.add_trace(go.Surface(
        z=mean_E,
        x=vision_vals,
        y=movement_vals,
        colorscale='Blues',
        name='Eating Synchrony'
    ))

    fig.add_trace(go.Surface(
        z=mean_R,
        x=vision_vals,
        y=movement_vals,
        colorscale='Reds',
        name='Resting Synchrony',
        opacity=0.7
    ))

    fig.update_layout(
        title=f"{n_cows}-Cow Herd — {TOPOLOGY_LABELS.get(topology, topology)} Topology",
        scene=dict(
            xaxis_title="Vision Radius",
            yaxis_title="Movement Step Size",
            zaxis_title="Mean Synchrony Error (minutes)"
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    if filename is None:
        filename = f"synchrony_vizmvmt_{topology}_{n_cows}cows.html"
    fig.write_html(filename)



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


def compute_pairwise_synchrony(times_i, times_j, max_shift=10):
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

        delta = np.mean(np.abs(np.array(cropped_i) - np.array(cropped_j))) * (0.5 / 60)
        best_delta = min(best_delta, delta)

    return best_delta

def compute_herd_synchrony(state_sequences, max_shift=10, verbose: bool = True):
    """
    Computes average herd synchrony over all pairs (i, j) including i = j,
    using Equation (31) from Sun et al. (2022).

    Parameters:
        state_sequences: list of observable state arrays (1 per cow)
        max_shift: maximum time shift allowed when aligning transitions

    Returns:
        (avg_delta_E, avg_delta_R, total_delta) — all in minutes
    """
    n = len(state_sequences)
    tau_list = [get_transition_times(states, 0) for states in state_sequences]
    kappa_list = [get_transition_times(states, 1) for states in state_sequences]
    if verbose: print("Transition counts per cow:")
    if verbose: print("  Eating :", [len(t) for t in tau_list])
    if verbose: print("  Resting:", [len(k) for k in kappa_list])

    total_delta_E = 0
    total_delta_R = 0

    for i in range(n):
        for j in range(n):
            if i == j:
                delta_E = 0.0
                delta_R = 0.0
            else:
                delta_E = compute_pairwise_synchrony(tau_list[i], tau_list[j], max_shift)
                delta_R = compute_pairwise_synchrony(kappa_list[i], kappa_list[j], max_shift)

            total_delta_E += delta_E
            total_delta_R += delta_R

    normalizer = n * n
    avg_delta_E = total_delta_E / normalizer
    avg_delta_R = total_delta_R / normalizer
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

# ============================
# State Analysis Utilities
# ============================
def compute_state_proportions(state_sequences):
    """
    Computes the proportion of time each cow spends in Eating, Resting, and Standing states.

    Parameters:
        state_sequences (list of lists): Each inner list contains the state sequence (0=E, 1=R, 2=S) for one cow.

    Returns:
        tuple of np.ndarray: Three arrays of shape (n_cows,) corresponding to the proportion of time in
                             Eating (E), Resting (R), and Standing (S) states, respectively.
    """
    n_cows = len(state_sequences)
    total_time = len(state_sequences[0])
    counts = np.zeros((n_cows, 3))  # E, R, S

    for i, seq in enumerate(state_sequences):
        for state in [0, 1, 2]:
            counts[i, state] = np.sum(np.array(seq) == state)

    props = counts / total_time
    return props[:, 0], props[:, 1], props[:, 2]  # E, R, S

def choose_adjacency(topology, n_cows, rng=None):
    """
    Builds an adjacency matrix representing cow interactions based on the specified network topology.

    Parameters
    ----------
    topology : str
        The type of topology to use. Supported options:
        - 'full': Fully connected network (complete graph)
        - 'grid': 2x5 grid layout (expects n_cows == 10)
        - 'random_p03': Random network with edge probability 0.3
        - 'random_p075': Random network with edge probability 0.75
    n_cows : int
        Number of cows in the herd.
    rng : np.random.Generator, optional
        A random number generator instance. If None, a new default generator is used.

    Returns
    -------
    A : np.ndarray
        An (n_cows x n_cows) adjacency matrix with 1s for connections and 0s otherwise.

    Raises
    ------
    AssertionError
        If `topology` is 'grid' but `n_cows` is not 10.
    ValueError
        If `topology` is not recognized.
    """
    if topology == 'full':
        A = np.ones((n_cows, n_cows)) - np.eye(n_cows)
    elif topology == 'grid':
        assert n_cows == 10, "Grid currently expects 10 cows"
        A = build_grid_adjacency(rows=2, cols=5)
    elif topology.startswith('random_p'):
        if rng is None:
            rng = np.random.default_rng()
        if topology == 'random_p03':
            edge_prob = 0.3
        elif topology == 'random_p075':
            edge_prob = 0.75
        else:
            raise ValueError(f"Unrecognized random topology: {topology}")
        A = rng.random((n_cows, n_cows))
        A = np.triu(A, 1) < edge_prob
        A = A.astype(int)
        A = A + A.T
        np.fill_diagonal(A, 0)
    else:
        raise ValueError(f"Unknown topology: {topology}")
    return A