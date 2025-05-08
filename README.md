# Cow Synchronization Model (APMA 1360 Final Project)

This repository contains code and data for our APMA 1360 final project on the synchronization behavior of cows modeled using hybrid dynamical systems. Inspired by Sun & Holmes (2022), we extend their work by simulating cow interactions in various network topologies and exploring emergent synchronization under different coupling conditions.

## Contents

- `main.py`: Entry point for running periodic orbit simulations or herd synchrony experiments.
- `cow.py`: Defines the behavior of a single cow.
- `cowherd.py`: Manages multi-cow simulation and coupling structure.
- `simulation.py`: Contains the main simulation logic for periodic orbits and herds.
- `utils.py`: Helper functions for plotting, metrics, and adjacency matrices.
- `walking_sim.py`: An experimental extension that simulates cow movement on a grid.
- `csvs/`: Contains generated CSV files with trial-level synchrony metrics.
- `figures/`: Contains all figures used in the final report and graphs of each periodic orbit (cases A–D).
- `Reference_Material_for_Cow_Synchronization.pdf`: A self-contained reference sheet summarizing the model from Sun & Holmes (2022), including key equations, switching rules, and coupling structure.

## How to Run

This project uses Python 3.10+ and the following dependencies:

```bash
pip install numpy pandas matplotlib
```

To run simulations and generate plots:
```bash
python main.py
```
By default, main.py runs the full herd synchronization experiment (run_herd_synchrony_experiment) across four network topologies. This process is compute-intensive (∼6 hours total). You may toggle the relevant flags in main.py to instead run selected periodic orbit cases (A–D).

To run the walking cow simulation separately:

```bash
python walking_sim.py
```

## Reproducing Our Results

- Periodic orbit plots (Cases A–D) are generated using simulate_periodic_orbit_* functions in simulation.py.
- Synchronization plots are saved to figures/ and CSV stats are written to csvs/.
- The main.py file controls both experiment types — edit or uncomment relevant lines to control the run.

## Citation

If referencing this work, please cite:
```bibtex
@misc{apma1360_final_project,
  author       = {Khurana, A. and Collaborators},
  title        = {APMA 1360 Final Project: Cow Synchronization Models},
  year         = {2025},
  howpublished = {\url{https://github.com/brown-afkhurana/apma1360-final}},
  note         = {Accessed: 2025-05-08}
}
```

## License

This code is provided for academic and instructional purposes only.
