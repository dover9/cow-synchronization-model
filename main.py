import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from cow import Cow
from cowherd import CowHerd
from simulation import (
    simulate_periodic_orbit_A,
    simulate_periodic_orbit_B,
    simulate_periodic_orbit_C,
    simulate_periodic_orbit_D,
    simulate_one_cow,
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

def compute_state_proportions(state_sequences):
    n_cows = len(state_sequences)
    total_time = len(state_sequences[0])
    counts = np.zeros((n_cows, 3))  # E, R, S

    for i, seq in enumerate(state_sequences):
        for state in [0, 1, 2]:
            counts[i, state] = np.sum(np.array(seq) == state)

    props = counts / total_time
    return props[:, 0], props[:, 1], props[:, 2]  # E, R, S

def choose_adjacency(topology, n_cows, rng=None):
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


def run_herd_synchrony_experiment():
    sigma_vals = np.linspace(0.00, 0.05, 50)
    n_trials = 50
    n_cows = 10
    param_noise = 0.001
    timesteps = 30000
    stepsize = 0.5
    delta = 0.25

    master_rng = np.random.default_rng(seed=42)

    trial_stats = {
        "sigma": [],
        "topology": [],
        "n_cows": [],
        "trial": [],
        "mean_transitions_e": [],
        "mean_transitions_r": [],
        "mean_time_e": [],
        "mean_time_r": [],
        "prop_e": [],
        "prop_r": [],
        "prop_s": []
    }

    # for topology in ['full', 'grid', 'random_p03', 'random_p075']:
    # for topology in ['grid', 'random_p03', 'random_p075']:
    for topology in ['random_p075']:
        print(f"Starting topology: {topology}")
        mean_E = []
        std_E = []
        mean_R = []
        std_R = []

        for sigma in sigma_vals:
            print(f"  σ = {sigma:.4f}")
            deltas_E = []
            deltas_R = []

            for trial_num in range(n_trials):
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

                e_trans = [get_transition_times(seq, 0) for seq in herd]
                r_trans = [get_transition_times(seq, 1) for seq in herd]

                trans_counts_e = [len(t) for t in e_trans]
                trans_counts_r = [len(t) for t in r_trans]

                prop_e, prop_r, prop_s = compute_state_proportions(herd)

                mean_time_e = np.mean(np.diff(e_trans[0])) if len(e_trans[0]) > 1 else np.nan
                mean_time_r = np.mean(np.diff(r_trans[0])) if len(r_trans[0]) > 1 else np.nan

                trial_stats["sigma"].append(sigma)
                trial_stats["topology"].append(topology)
                trial_stats["n_cows"].append(n_cows)
                trial_stats["trial"].append(trial_num)
                trial_stats["mean_transitions_e"].append(np.mean(trans_counts_e))
                trial_stats["mean_transitions_r"].append(np.mean(trans_counts_r))
                trial_stats["mean_time_e"].append(mean_time_e)
                trial_stats["mean_time_r"].append(mean_time_r)
                trial_stats["prop_e"].append(np.mean(prop_e))
                trial_stats["prop_r"].append(np.mean(prop_r))
                trial_stats["prop_s"].append(np.mean(prop_s))

                delta_E, delta_R, _ = compute_herd_synchrony(herd)
                if np.isfinite(delta_E):
                    deltas_E.append(delta_E)
                if np.isfinite(delta_R):
                    deltas_R.append(delta_R)

            print(f"  σ = {sigma:.4f}: kept {len(deltas_E)} eating, {len(deltas_R)} resting trials")

            mean_E.append(np.mean(deltas_E))
            std_E.append(np.std(deltas_E))
            mean_R.append(np.mean(deltas_R))
            std_R.append(np.std(deltas_R))

        print(f"{topology.capitalize()} topology Δ^R: min={np.min(mean_R):.3f}, max={np.max(mean_R):.3f}")
        plot_synchrony_vs_sigma(
            sigma_vals,
            mean_E, std_E,
            mean_R, std_R,
            topology,
            n_cows
        )

        # Save stats to CSV
        df = pd.DataFrame(trial_stats)
        df.to_csv(f"herd_stats_{topology}_{n_cows}cows.csv", index=False)
        trial_stats = {key: [] for key in trial_stats}  # reset for next topology


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

    # Set this to True only when you're ready
    RUN_HERD_EXPERIMENT = True

    if RUN_HERD_EXPERIMENT:
        start = time.time()
        run_herd_synchrony_experiment()
        print(f"Total time: {time.time() - start:.1f} seconds")