import numpy as np
from cow import Cow

class CowHerd:
    """
    Represents a herd of interacting cows with coupled dynamics.

    This class manages multiple Cow instances, allowing for their hidden states
    to evolve under both intrinsic dynamics and neighbor-induced coupling as 
    described in Equation (26) of the "Synchronization of Cows" paper.

    Attributes:
        cows: List of Cow objects representing individual cows in the herd.
        adjacency: Adjacency matrix (numpy array) specifying the interaction network.
        sigma_x: Coupling strength affecting hunger (x) dynamics.
        sigma_y: Coupling strength affecting tiredness (y) dynamics.
    """
    def __init__(self, cows: list[Cow], adjacency_matrix: np.ndarray, sigma_x: float = 0.05, sigma_y: float = 0.05):
        """
        Initialize a CowHerd instance.

        Args:
            cows: List of Cow instances.
            adjacency_matrix: 2D numpy array indicating pairwise connections between cows.
                Entry (i, j) = 1 if cow i observes cow j; 0 otherwise.
            sigma_x: Coupling strength for the x-dynamics (default 0.05).
            sigma_y: Coupling strength for the y-dynamics (default 0.05).
        """
        self.cows = cows
        self.adjacency = adjacency_matrix
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

    def step(self, stepsize: float = 0.01):
        """
        Advance the herd dynamics by one discrete time step.

        For each cow:
            - Compute the effective drift rates modified by neighboring cows' states.
            - Update the hidden states (x, y) via Euler integration.
            - Update the observable state based on thresholds.

        Args:
            stepsize: Size of the Euler integration time step (default 0.01).
        """
        # Get neighbors' observable states for each cow
        neighbor_states_list = []
        for i, cow in enumerate(self.cows):
            neighbors = [j for j in range(len(self.cows)) if self.adjacency[i, j]]
            neighbor_states = [self.cows[j].obs_state for j in neighbors]
            neighbor_states_list.append(neighbor_states)

        # Update hidden states using coupled derivatives
        for cow, neighbor_states in zip(self.cows, neighbor_states_list):
            dx, dy = cow.coupled_hidden_state_derivs(neighbor_states, self.sigma_x, self.sigma_y)
            cow.x += stepsize * dx
            cow.y += stepsize * dy
            # Clamp hidden states to [0, 1] to maintain physiological realism and prevent numerical drift
            cow.x = np.clip(cow.x, 0, 1)
            cow.y = np.clip(cow.y, 0, 1)

        # Update observable states
        for cow in self.cows:
            cow.next_obs_state()