# Cow Synchronization Model (APMA 1360 Final Project)

This repository contains code and data for our APMA 1360 final project on modeling synchronization behavior in cows using hybrid dynamical systems. Inspired by Sun & Holmes (2022), the project simulates cow interactions across several network topologies and investigates how coupling strength influences emergent group behavior.

## Project Summary

Our goal was to understand how simple individual rules and network structure can lead to synchronization in multi-agent systems. We implemented a hybrid dynamical model of cow behavior and explored both two-cow and multi-cow setups using Python. The project combines ideas from nonlinear dynamics, numerical simulation, and data analysis.

### My Role

This group project was completed as part of APMA 1360 at Brown University. Collaborators included Tuongvi Victoria Vo, Thomas Wang, and Arjun Khurana. My primary contributions included:
- Writing the simulation and coupling logic in `cowherd.py` and `simulation.py`
- Designing and implementing the main experiment flow (`main.py`)
- Managing output organization and plotting scripts
- Co-writing the final report and editing visualizations

## Contents

- `main.py`: Entry point for running periodic orbit simulations or herd synchrony experiments.
- `cow.py`: Defines the behavior of a single cow.
- `cowherd.py`: Manages multi-cow simulation and coupling structure.
- `simulation.py`: Contains core simulation logic.
- `utils.py`: Helper functions for plotting, metrics, and adjacency matrices.
- `walking_sim.py`: Simulates cow movement on a 2D grid.
- `csvs/`: Contains trial-level output metrics.
- `figures/`: Stores plots used in the final report.
- `Reference_Material_for_Cow_Synchronization.pdf`: Annotated reference summary of the Sun & Holmes model.

## Final Report

For full details on the modeling approach, simulations, and results, see our [final project report](./Cow_Synchronization_Final_Report.pdf), submitted for APMA 1360 at Brown University.

## How to Run

This project requires Python 3.10+ and:
```bash
pip install numpy pandas matplotlib
```
To run the full herd synchronization experiment across all topologies (∼6 hours total):
```bash
python main.py
```
To test smaller components (e.g., two-cow or periodic orbits), open `main.py`, and comment/uncomment the appropriate function calls.

To run the experimental walking cow simulation:
```bash
python walking_sim.py
```
## Reproducing Our Results

- Periodic orbit plots (Cases A–D) are generated using `simulate_periodic_orbit_*` functions in `simulation.py`.
- Synchrony metrics and plots are saved manually to `csvs/` and `figures/` after each run.
- `main.py` controls which experiments are active—modify it directly to isolate behaviors.

## Citation

If referencing this work, please cite:
```bibtex
@misc{cow_synchronization_apma1360,
  author       = {Dover, Joshua and Collaborators},
  title        = {APMA 1360 Final Project: Cow Synchronization Model},
  year         = {2025},
  howpublished = {\url{https://github.com/dover9/cow-synchronization-model}},
  note         = {Accessed: 2025-05-10}
}
```

## License

This code is provided for academic and instructional purposes only.
