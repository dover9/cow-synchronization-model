import numpy as np
import matplotlib.pyplot as plt
from simulation import (
    simulate_periodic_orbit_A,
    simulate_periodic_orbit_B,
    simulate_periodic_orbit_C,
    simulate_periodic_orbit_D,
    simulate_two_cows,
    simulate_herd
    
)
from utils import (
    plot_single_cow,
    plot_observable_states,
    plot_synchrony_vs_sigma,
    get_transition_times,
    compute_pairwise_synchrony,
    compute_herd_synchrony,
    build_grid_adjacency,
    build_random_adjacency)

def choose_adjacency(topology, n_cows, seed=42):
    if topology == 'full':
        A = np.ones((n_cows, n_cows)) - np.eye(n_cows)
    elif topology == 'grid':
        side = int(np.sqrt(n_cows))
        assert side ** 2 == n_cows, "Grid topology requires a square number of cows"
        A = build_grid_adjacency(side, side)
    elif topology == 'random':
        np.random.seed(seed)
        A = np.random.rand(n_cows, n_cows)
        A = np.triu(A, 1)
        A = (A + A.T) < 0.3  # keep 30% of edges
        A = A.astype(int)
        np.fill_diagonal(A, 0)
    else:
        raise ValueError(f"Unknown topology: {topology}")
    return A

sigma_vals = np.linspace(0.001, 0.05, 40)
n_trials = 30
n_cows = 10

for topology in ['full', 'grid', 'random']:
    mean_E = []
    std_E = []
    mean_R = []
    std_R = []

    for sigma in sigma_vals:
        deltas_E = []
        deltas_R = []

        for _ in range(n_trials):
            A = choose_adjacency(topology, n_cows)
            herd = simulate_herd(n_cows, sigma_x=sigma, sigma_y=sigma, A=A)
            delta_E, delta_R, _ = compute_herd_synchrony(herd)
            deltas_E.append(delta_E)
            deltas_R.append(delta_R)

        mean_E.append(np.mean(deltas_E))
        std_E.append(np.std(deltas_E))
        mean_R.append(np.mean(deltas_R))
        std_R.append(np.std(deltas_R))

    plot_synchrony_vs_sigma(
        sigma_vals,
        mean_E, std_E,
        mean_R, std_R,
        title=f"{n_cows}-Cow Herd — {topology.capitalize()} Topology"
    )

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
    herd_states = [states_1, states_2]

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

    # states_1, states_2 = simulate_two_cows()
    # herd_states = [states_1, states_2]

    delta_E, delta_R, total = compute_herd_synchrony(herd_states)
    print(f"Herd Synchrony, Δ^E = {delta_E:.2f}, Δ^R = {delta_R:.2f}, Δ = {total:.2f}")

    herd_states = simulate_herd(n_cows=10)
    delta_E, delta_R, total = compute_herd_synchrony(herd_states)
    print(f"10-Cow Herd, Δ^E = {delta_E:.2f}, Δ^R = {delta_R:.2f}, Δ = {total:.2f}")

    # Select time slice
    start = 3000
    end = 3200
    states_1 = states_1[start:end]
    states_2 = states_2[start:end]

    # Plot 2-cow simulation
    # plot_observable_states(states_1, states_2, start=start, title="Observable States")

    A_rand = build_random_adjacency(n_cows=10, p=0.4, seed=42)
    herd_states = simulate_herd(n_cows=10, A=A_rand)
    delta_E, delta_R, total = compute_herd_synchrony(herd_states)
    print(f"Random Network, Δ^E = {delta_E:.2f}, Δ^R = {delta_R:.2f}, Δ = {total:.2f}")
    print(A_rand)