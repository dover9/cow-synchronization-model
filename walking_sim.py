import time

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from cow import Cow
from cowherd import CowHerd
from utils import get_transition_times, compute_state_proportions, compute_herd_synchrony, plot_synchrony_vs_mvmt, plot_synchrony_vs_sigma, plot_synchrony_vs_viz


class WalkingSim:
    def __init__(self,
                 cows: list[Cow],
                 cow_vision_range: float = 0.2,
                 cow_movement_range: float = 0.05):
        self.cows = cows
        self.cow_positions = []
        for cow in cows:
            # generate random initial positions
            x = np.random.uniform(0, 1)
            y = np.random.uniform(0, 1)
            self.cow_positions.append((x, y))
        self.cow_vision_range = cow_vision_range
        self.cow_movement_range = cow_movement_range

    def who_can_cow_see(self, cow_index: int) -> list[int]:
        """
        Returns a list of indices of cows that cow_index can see.
        """
        cow = self.cows[cow_index]
        x, y = self.cow_positions[cow_index]

        visible_cows = []
        for i, (other_x, other_y) in enumerate(self.cow_positions):
            if i == cow_index:
                continue
            # check if cow in vision range
            if np.sqrt((x - other_x) ** 2 + (y - other_y) ** 2) < self.cow_vision_range:
                visible_cows.append(i)
        return visible_cows
    
    def generate_adjacency_matrix(self):
        """
        Generates an adjacency matrix based on the visibility of cows.
        """
        n_cows = len(self.cows)
        A = np.zeros((n_cows, n_cows))
        for i in range(n_cows):
            visible_cows = self.who_can_cow_see(i)
            for j in visible_cows:
                A[i, j] = 1
        return A
        
    def update_positions(self):
        for i, cow in enumerate(self.cows):
            # check if cow is standing
            if cow.obs_state != "S":
                continue
            # update cow position
            x, y = self.cow_positions[i]
            self.cow_positions[i] = (
                x + np.random.uniform(-self.cow_movement_range, self.cow_movement_range),
                y + np.random.uniform(-self.cow_movement_range, self.cow_movement_range)
            )
            # ensure cow stays within bounds
            self.cow_positions[i] = (
                max(0, min(1, self.cow_positions[i][0])),
                max(0, min(1, self.cow_positions[i][1]))
            )


def test_walking_sim():
    """Test walking, without cow states."""
    cows = [Cow() for _ in range(5)]
    for cow in cows:
        cow.obs_state = "S"
    sim = WalkingSim(cows)
    # record cow positions over time
    positions_over_time = []
    for _ in range(100):
        sim.update_positions()
        positions_over_time.append(sim.cow_positions.copy())
    print(len(positions_over_time))
    # plot the positions
    for i in range(len(cows)):
        x = [pos[i][0] for pos in positions_over_time]
        y = [pos[i][1] for pos in positions_over_time]
        plt.plot(x, y, label=f"Cow {i}")
    plt.title("Cow Positions Over Time")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.show()


def simulate_walking_cowherd(n_cows: int, n_steps: int):
    cows = [Cow(params=(0.05, 0.1, 0.05, 0.125)) for _ in range(n_cows)]
    sim = WalkingSim(cows)
    herd = CowHerd(cows,
                   sim.generate_adjacency_matrix())
    positions_over_time = []
    for i in range(n_steps):
        herd.adjacency = sim.generate_adjacency_matrix()
        herd.step()
        sim.update_positions()
        positions_over_time.append(sim.cow_positions.copy())
    # plot
    for i in range(n_cows):
        x = [pos[i][0] for pos in positions_over_time]
        y = [pos[i][1] for pos in positions_over_time]
        plt.plot(x, y, label=f"Cow {i}")
    plt.title("Cow Positions Over Time")


def simulate_herd(n_cows=10, 
                  timesteps=30000, 
                  stepsize=0.5,
                  sigma_x=0.05, 
                  sigma_y=0.05,
                  param_base=(0.05, 0.1, 0.05, 0.125),
                  param_noise=0.001,
                  init_state=(1.0, 0.1),
                  delta=0.25,
                  cow_vision_range=0.2,
                  cow_movement_range=0.05):
    """
    Copied from simulation.py, adapted to use walking.
    """
    base = np.array(param_base)
    cows = []

    for _ in range(n_cows):
        # Add slight noise to parameters
        noise = np.random.uniform(-param_noise, param_noise, size=4)
        params = tuple(base + noise)
        cow = Cow(params=params, init_obsstate="E", init_hiddenstate=init_state, delta=delta)
        cows.append(cow)

    sim = WalkingSim(cows,
                     cow_vision_range=cow_vision_range,
                     cow_movement_range=cow_movement_range)

    # Create Herd
    herd = CowHerd(cows, 
                   sim.generate_adjacency_matrix(), 
                   sigma_x=sigma_x, 
                   sigma_y=sigma_y)

    state_history = [[] for _ in range(n_cows)]
    position_history = [[] for _ in range(n_cows)]

    for _ in range(timesteps):
        herd.adjacency = sim.generate_adjacency_matrix()
        herd.step(stepsize)
        sim.update_positions()
        for i, cow in enumerate(cows):
            state_history[i].append(["E", "R", "S"].index(cow.obs_state))
            position_history[i].append(sim.cow_positions[i])

    return state_history, position_history



def run_herd_synchrony_experiment_vary_sigma(verbose: bool = False):
    """
    Adapted from main.py.
    """
    sigma_vals = np.linspace(0.00, 0.05, 50)
    n_trials = 20
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

    topology = "WALKING_COWHERD"

    if verbose: print(f"Starting topology: {topology}")
    mean_E = []
    std_E = []
    mean_R = []
    std_R = []

    for sigma in tqdm(sigma_vals, desc="Sigma values", unit="sigma"):
        if verbose: print(f"  σ = {sigma:.4f}")
        deltas_E = []
        deltas_R = []

        for trial_num in tqdm(range(n_trials), desc="Trials", unit="trial", leave=False):
            trial_rng = np.random.default_rng(master_rng.integers(1e9))

            herd, herd_pos_hist = simulate_herd(
                n_cows=n_cows,
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

            delta_E, delta_R, _ = compute_herd_synchrony(herd, verbose=verbose)
            if np.isfinite(delta_E):
                deltas_E.append(delta_E)
            if np.isfinite(delta_R):
                deltas_R.append(delta_R)

        if verbose: print(f"  σ = {sigma:.4f}: kept {len(deltas_E)} eating, {len(deltas_R)} resting trials")

        mean_E.append(np.mean(deltas_E))
        std_E.append(np.std(deltas_E))
        mean_R.append(np.mean(deltas_R))
        std_R.append(np.std(deltas_R))

    print(f"{topology} topology Δ^R: min={np.min(mean_R):.3f}, max={np.max(mean_R):.3f}")
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


def run_herd_synchrony_experiment_vary_movement(verbose: bool = False):
    """
    Adapted from main.py.
    """
    sigma = 0.05
    n_trials = 20
    n_cows = 10
    param_noise = 0.001
    timesteps = 30000
    stepsize = 0.5
    delta = 0.25
    movement_vals = np.linspace(0.01, 0.1, 20)
    vision_val = 0.2

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
        "prop_s": [],
        "movement_range": [],
        "vision_range": []
    }

    topology = "WALKING_COWHERD"

    if verbose: print(f"Starting topology: {topology}")
    mean_E = []
    std_E = []
    mean_R = []
    std_R = []

    for movement_val in tqdm(movement_vals, desc="Movement values"):
        if verbose: print(f"  mv = {movement_val:.4f}")
        deltas_E = []
        deltas_R = []

        for trial_num in tqdm(range(n_trials), desc="Trials", unit="trial", leave=False):
            trial_rng = np.random.default_rng(master_rng.integers(1e9))

            herd, herd_pos_hist = simulate_herd(
                n_cows=n_cows,
                sigma_x=sigma,
                sigma_y=sigma,
                param_noise=param_noise,
                timesteps=timesteps,
                stepsize=stepsize,
                delta=delta,
                cow_movement_range=movement_val,
                cow_vision_range=vision_val,
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
            trial_stats["movement_range"].append(movement_val)
            trial_stats["vision_range"].append(vision_val)

            delta_E, delta_R, _ = compute_herd_synchrony(herd, verbose=verbose)
            if np.isfinite(delta_E):
                deltas_E.append(delta_E)
            if np.isfinite(delta_R):
                deltas_R.append(delta_R)

        if verbose: print(f"  mv = {movement_val:.4f}: kept {len(deltas_E)} eating, {len(deltas_R)} resting trials")

        mean_E.append(np.mean(deltas_E))
        std_E.append(np.std(deltas_E))
        mean_R.append(np.mean(deltas_R))
        std_R.append(np.std(deltas_R))

    print(f"{topology} topology Δ^R: min={np.min(mean_R):.3f}, max={np.max(mean_R):.3f}")


    plot_synchrony_vs_mvmt(
        movement_vals,
        mean_E, std_E,
        mean_R, std_R,
        topology,
        n_cows,
        f"synchrony_WALKING_{n_cows}cows_varymvmt.png"
    )

    # Save stats to CSV
    df = pd.DataFrame(trial_stats)
    df.to_csv(f"herd_stats_WALKING_{n_cows}cows_varymvmt.csv", index=False)
    trial_stats = {key: [] for key in trial_stats}  # reset for next topology


def run_herd_synchrony_experiment_vary_vision(verbose: bool = False):
    """
    Adapted from main.py.
    """
    sigma = 0.05
    n_trials = 20
    n_cows = 10
    param_noise = 0.001
    timesteps = 30000
    stepsize = 0.5
    delta = 0.25
    movement_val = 0.02
    vision_vals = np.linspace(0.01, 0.4, 25)

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
        "prop_s": [],
        "movement_range": [],
        "vision_range": []
    }

    topology = "WALKING_COWHERD"

    if verbose: print(f"Starting topology: {topology}")
    mean_E = []
    std_E = []
    mean_R = []
    std_R = []

    for vision_val in tqdm(vision_vals, desc="Movement values"):
        if verbose: print(f"  mv = {movement_val:.4f}")
        deltas_E = []
        deltas_R = []

        for trial_num in tqdm(range(n_trials), desc="Trials", unit="trial", leave=False):
            trial_rng = np.random.default_rng(master_rng.integers(1e9))

            herd, herd_pos_hist = simulate_herd(
                n_cows=n_cows,
                sigma_x=sigma,
                sigma_y=sigma,
                param_noise=param_noise,
                timesteps=timesteps,
                stepsize=stepsize,
                delta=delta,
                cow_movement_range=movement_val,
                cow_vision_range=vision_val,
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
            trial_stats["movement_range"].append(movement_val)
            trial_stats["vision_range"].append(vision_val)

            delta_E, delta_R, _ = compute_herd_synchrony(herd, verbose=verbose)
            if np.isfinite(delta_E):
                deltas_E.append(delta_E)
            if np.isfinite(delta_R):
                deltas_R.append(delta_R)

        if verbose: print(f"  vv = {vision_val:.4f}: kept {len(deltas_E)} eating, {len(deltas_R)} resting trials")

        mean_E.append(np.mean(deltas_E))
        std_E.append(np.std(deltas_E))
        mean_R.append(np.mean(deltas_R))
        std_R.append(np.std(deltas_R))

    print(f"{topology} topology Δ^R: min={np.min(mean_R):.3f}, max={np.max(mean_R):.3f}")


    plot_synchrony_vs_viz(
        vision_vals,
        mean_E, std_E,
        mean_R, std_R,
        topology,
        n_cows,
        f"synchrony_WALKING_{n_cows}cows_varyviz.png"
    )

    # Save stats to CSV
    df = pd.DataFrame(trial_stats)
    df.to_csv(f"herd_stats_WALKING_{n_cows}cows_varyviz.csv", index=False)
    trial_stats = {key: [] for key in trial_stats}  # reset for next topology


if __name__ == "__main__":
    start = time.time()
    run_herd_synchrony_experiment_vary_vision(verbose=False)
    end = time.time()
    print(f"Total time: {end - start:.2f} seconds")
