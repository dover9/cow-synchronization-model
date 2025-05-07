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
    simulate_herd  
)
from utils import (
    plot_single_cow,
    plot_synchrony_vs_sigma,
    get_transition_times,
    compute_herd_synchrony,
    build_grid_adjacency,
    compute_state_proportions,
    choose_adjacency
    )

def run_periodic_orbit(case=None):
    """
    Simulates and plots periodic orbits for one or all cow behavior cases.

    Parameters:
        case (str or None): If None, runs all predefined periodic orbit simulations (A–D).
                            If a single character (e.g., "A", "B", "C", "D"), only that case is simulated and plotted.

    Raises:
        ValueError: If an invalid case label is provided.
    """
    periodic_cases = {
        "A": simulate_periodic_orbit_A,
        "B": simulate_periodic_orbit_B,
        "C": simulate_periodic_orbit_C,
        "D": simulate_periodic_orbit_D,
    }

    if case is None:
        for label, sim_func in periodic_cases.items():
            hidden, obs, switches = sim_func()
            plot_single_cow(hidden, obs, switches, case_label=label)
    else:
        case = case.upper()
        if case not in periodic_cases:
            raise ValueError(f"Invalid case '{case}'. Valid options are: {list(periodic_cases.keys())}")
        hidden, obs, switches = periodic_cases[case]()
        plot_single_cow(hidden, obs, switches, case_label=case)


def run_herd_synchrony_experiment():
    """
    Runs a series of herd simulations across a range of coupling strengths (sigma values)
    and computes synchronization metrics under different network topologies.

    The function simulates multiple trials for each sigma value, measures herd synchrony,
    tracks transition statistics and state proportions, and saves the results to CSV files.
    It also generates plots of synchrony (Δ^E and Δ^R) versus sigma.

    Parameters
    ----------
    None

    Outputs
    -------
    - Plots of synchronization vs sigma for each topology.
    - CSV file per topology containing trial-level statistics.
    """
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

    for topology in ['full', 'grid', 'random_p03', 'random_p075']:
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
    # Run all periodic orbits
    run_periodic_orbit()

    # Or run just one
    # run_periodic_orbit("B")

    # Set this to True only when you're ready 
    # 10 cows @ 30,000 steps/0.5 step size takes 90 minutes per topology (there are 4 topologies)
    RUN_HERD_EXPERIMENT = True

    if RUN_HERD_EXPERIMENT:
        start = time.time()
        run_herd_synchrony_experiment()
        print(f"Total time: {time.time() - start:.1f} seconds")