import numpy as np
import matplotlib.pyplot as plt

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