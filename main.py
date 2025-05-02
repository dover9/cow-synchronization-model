import numpy as np
import matplotlib.pyplot as plt
from simulation import (
    simulate_periodic_orbit_A,
    simulate_periodic_orbit_B,
    simulate_periodic_orbit_C,
    simulate_periodic_orbit_D,
    simulate_two_cows
)
from utils import plot_single_cow, plot_observable_states

def get_transition_times(obs_states, target_state):
    """
    Returns the list of time indices where the cow transitions into `target_state`.
    obs_states: list or array of observable states (0 = E, 1 = R, 2 = S)
    target_state: integer (0, 1, or 2)
    """
    return [
        t for t in range(1, len(obs_states))
        if obs_states[t] == target_state and obs_states[t - 1] != target_state
    ]

def compute_pairwise_synchrony(times_i, times_j, max_shift=10):
    """
    Computes the minimum average absolute difference between two time series,
    allowing for integer shifts in alignment.

    Parameters:
        times_i, times_j: Lists of time steps (e.g., when cow i and j enter Eating)
        max_shift: Maximum number of steps to shift for alignment

    Returns:
        Minimum average absolute difference (float)
    """
    min_len = min(len(times_i), len(times_j))
    # No events; can't measure synchrony
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

        delta = np.mean(np.abs(np.array(cropped_i) - np.array(cropped_j)))
        best_delta = min(best_delta, delta)

    return best_delta

if __name__ == "__main__":
    # # Plot periodic orbit for case A
    # hidden, obs, switches = simulate_periodic_orbit_A()
    # plot_single_cow(hidden, obs, switches, case_label="A")

    # # Plot periodic orbit for case B
    # hidden, obs, switches = simulate_periodic_orbit_B()
    # plot_single_cow(hidden, obs, switches, case_label="B")

    # # Plot periodic orbit for case C
    # hidden, obs, switches = simulate_periodic_orbit_C()
    # plot_single_cow(hidden, obs, switches, case_label="C")

    # # Plot periodic orbit for case D
    # hidden, obs, switches = simulate_periodic_orbit_D()
    # plot_single_cow(hidden, obs, switches, case_label="D")

    # Test: Perfect synchrony
    tau_1 = [100, 200, 300, 400]
    tau_2 = [100, 200, 300, 400]

    delta_E = compute_pairwise_synchrony(tau_1, tau_2)
    print(f"[Perfect match] Δ^E = {delta_E:.2f}")  # Expect 0.00

    # Test: Constant offset
    tau_1 = [100, 200, 300, 400]
    tau_2 = [105, 205, 305, 405]

    delta_E = compute_pairwise_synchrony(tau_1, tau_2)
    print(f"[Offset by 5] Δ^E = {delta_E:.2f}")  # Expect ~5.00

    # Test: Random
    tau_1 = [100, 200, 300, 400]
    tau_2 = [130, 180, 290, 410]

    delta_E = compute_pairwise_synchrony(tau_1, tau_2)
    print(f"[Random offset] Δ^E = {delta_E:.2f}")  # Expect > 10

    # 2-cow simulation
    states_1, states_2 = simulate_two_cows()

    # Cow 1
    tau_1 = get_transition_times(states_1, target_state=0)  # Into Eating
    kappa_1 = get_transition_times(states_1, target_state=1)  # Into Resting

    # Cow 2
    tau_2 = get_transition_times(states_2, target_state=0)
    kappa_2 = get_transition_times(states_2, target_state=1)

    delta_E = compute_pairwise_synchrony(tau_1, tau_2)
    delta_R = compute_pairwise_synchrony(kappa_1, kappa_2)
    delta = delta_E + delta_R

    print(f"Δ^E = {delta_E:.2f}, Δ^R = {delta_R:.2f}, Total Δ = {delta:.2f}")   

    # Select time slice
    start = 3000
    end = 3200
    states_1 = states_1[start:end]
    states_2 = states_2[start:end]

    # Plot 2-cow simulation
    # plot_observable_states(states_1, states_2, start=start, title="Observable States")