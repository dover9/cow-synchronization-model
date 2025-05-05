import numpy as np
import matplotlib.pyplot as plt
from cow import Cow
from cowherd import CowHerd
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
    build_random_adjacency
    )

def choose_adjacency(topology, n_cows, rng=None):
    if topology == 'full':
        A = np.ones((n_cows, n_cows)) - np.eye(n_cows)
    elif topology == 'grid':
        # Manually define grid shape for consistent structure
        assert n_cows == 10, "Grid currently expects 10 cows"
        A = build_grid_adjacency(rows=2, cols=5)  # Or rows=5, cols=2 for vertical
    elif topology == 'random':
        if rng is None:
            rng = np.random.default_rng()
        edge_prob = 0.75
        A = rng.random((n_cows, n_cows))
        A = np.triu(A, 1) < edge_prob  # upper triangle only
        A = A.astype(int)
        A = A + A.T  # make symmetric
        np.fill_diagonal(A, 0)  # no self-connections
    else:
        raise ValueError(f"Unknown topology: {topology}")
    return A 

def run_herd_synchrony_experiment():
    sigma_vals = np.linspace(0.00, 0.05, 50)
    n_trials = 50
    n_cows = 10
    param_noise = 0.001
    timesteps = 30000
    stepsize = 0.5
    delta = 0.25

    master_rng = np.random.default_rng(seed=42)

    for topology in ['random']: #['full', 'grid', 'random']:
        print(f"Starting topology: {topology}")
        mean_E = []
        std_E = []
        mean_R = []
        std_R = []

        for sigma in sigma_vals:
            print(f"  σ = {sigma:.4f}")
            deltas_E = []
            deltas_R = []

            for _ in range(n_trials):
                trial_rng = np.random.default_rng(master_rng.integers(1e9))
                A = choose_adjacency(topology, n_cows, rng=trial_rng)

                herd = simulate_herd(
                    n_cows=n_cows,
                    A=A,
                    sigma_x=sigma,
                    sigma_y=sigma,
                    param_noise=param_noise,
                    timesteps=timesteps,
                    stepsize=stepsize,
                    delta=delta
                )

                delta_E, delta_R, _ = compute_herd_synchrony(herd)
                deltas_E.append(delta_E)
                deltas_R.append(delta_R)

            mean_E.append(np.mean(deltas_E))
            std_E.append(np.std(deltas_E))
            mean_R.append(np.mean(deltas_R))
            std_R.append(np.std(deltas_R))

        print(f"{topology.capitalize()} topology Δ^R: min={np.min(mean_R):.3f}, max={np.max(mean_R):.3f}")
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

    # # Test: Perfect synchrony
    # tau_1 = [100, 200, 300, 400]
    # tau_2 = [100, 200, 300, 400]

    # delta_E = compute_pairwise_synchrony(tau_1, tau_2)
    # print(f"[Perfect match] Δ^E = {delta_E:.2f}")  # Expect 0.00

    # # Test: Constant offset
    # tau_1 = [100, 200, 300, 400]
    # tau_2 = [105, 205, 305, 405]

    # delta_E = compute_pairwise_synchrony(tau_1, tau_2)
    # print(f"[Offset by 5] Δ^E = {delta_E:.2f}")  # Expect ~5.00

    # # Test: Random
    # tau_1 = [100, 200, 300, 400]
    # tau_2 = [130, 180, 290, 410]

    # delta_E = compute_pairwise_synchrony(tau_1, tau_2)
    # print(f"[Random offset] Δ^E = {delta_E:.2f}")  # Expect > 10

    # # 2-cow simulation
    # states_1, states_2 = simulate_two_cows()
    # herd_states = [states_1, states_2]

    # # Cow 1
    # tau_1 = get_transition_times(states_1, target_state=0)  # Into Eating
    # kappa_1 = get_transition_times(states_1, target_state=1)  # Into Resting

    # # Cow 2
    # tau_2 = get_transition_times(states_2, target_state=0)
    # kappa_2 = get_transition_times(states_2, target_state=1)

    # delta_E = compute_pairwise_synchrony(tau_1, tau_2)
    # delta_R = compute_pairwise_synchrony(kappa_1, kappa_2)
    # delta = delta_E + delta_R

    # print(f"Δ^E = {delta_E:.2f}, Δ^R = {delta_R:.2f}, Total Δ = {delta:.2f}")

    # states_1, states_2 = simulate_two_cows()
    # herd_states = [states_1, states_2]

    # delta_E, delta_R, total = compute_herd_synchrony(herd_states)
    # print(f"Herd Synchrony, Δ^E = {delta_E:.2f}, Δ^R = {delta_R:.2f}, Δ = {total:.2f}")

    # herd_states = simulate_herd(n_cows=10)
    # delta_E, delta_R, total = compute_herd_synchrony(herd_states)
    # print(f"10-Cow Herd, Δ^E = {delta_E:.2f}, Δ^R = {delta_R:.2f}, Δ = {total:.2f}")

    # # Select time slice
    # start = 3000
    # end = 3200
    # states_1 = states_1[start:end]
    # states_2 = states_2[start:end]

    # # Plot 2-cow simulation
    # plot_observable_states(states_1, states_2, start=start, title="Observable States")

    # plot_synchrony_vs_sigma(epsilon=0.001, n_trials=50)
    # plot_synchrony_vs_sigma(epsilon=0.01, n_trials=50)

    # Set this to True only when you're ready
    RUN_HERD_EXPERIMENT = True

    if RUN_HERD_EXPERIMENT:
        run_herd_synchrony_experiment()