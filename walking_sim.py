import numpy as np
import matplotlib.pyplot as plt

from cow import Cow
from cowherd import CowHerd


class WalkingSim:
    def __init__(self,
                 cows: list[Cow]):
        self.cows = cows
        self.cow_positions = []
        for cow in cows:
            # generate random initial positions
            x = np.random.uniform(0, 1)
            y = np.random.uniform(0, 1)
            self.cow_positions.append((x, y))
        self.cow_vision_range = 0.2
        self.cow_movement_range = 0.05

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

if __name__ == "__main__":
    simulate_walking_cowherd(5, 10000)
    plt.show()
