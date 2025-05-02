import numpy as np
import matplotlib.pyplot as plt
from cow import Cow
from cowherd import CowHerd
from simulation import (
    simulate_periodic_orbit_A,
    simulate_periodic_orbit_B,
    simulate_periodic_orbit_C,
    simulate_periodic_orbit_D,
    simulate_two_cows
)
from utils import plot_single_cow, plot_observable_states

if __name__ == "__main__":
    # Plot periodic orbit for case A
    hidden, obs, switches = simulate_periodic_orbit_A()
    plot_single_cow(hidden, obs, switches, case_label="A")

    # Plot periodic orbit for case B
    hidden, obs, switches = simulate_periodic_orbit_B()
    plot_single_cow(hidden, obs, switches, case_label="B")

    # Plot periodic orbit for case C
    hidden, obs, switches = simulate_periodic_orbit_C()
    plot_single_cow(hidden, obs, switches, case_label="C")

    # Plot periodic orbit for case D
    # Plot periodic orbit for case C
    hidden, obs, switches = simulate_periodic_orbit_D()
    plot_single_cow(hidden, obs, switches, case_label="D")

    full_states_1, full_states_2 = simulate_two_cows()

    # Select time slice
    start = 3000
    end = 3200
    states_1 = full_states_1[start:end]
    states_2 = full_states_2[start:end]

    # Plot 2-cow simulation
    plot_observable_states(states_1, states_2, start=start, title="Observable States")