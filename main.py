import numpy as np
import matplotlib.pyplot as plt


class Cow:
    """One cow. At this point covers paper up to but not including 2.3"""

    def __init__(self,
                 init_hiddenstate: tuple[float, float] = None,
                 init_obsstate: str = None,
                 params: tuple[float, float, float, float] = None,
                 delta: float = 0.01):
        if init_hiddenstate is None:
            # self.x = np.random.uniform(0, 1)
            # self.y = np.random.uniform(0, 1)
            self.x = 1
            self.y = np.random.uniform(delta, 1)
        else:
            self.x, self.y = init_hiddenstate

        if init_obsstate is None:
            # self.obs_state = np.random.choice(["E", "R", "S"])
            self.obs_state = "E"
        else:
            self.obs_state = init_obsstate

        if params is None:
            self.alpha_1 = np.random.uniform(0, 0.1)
            self.alpha_2 = np.random.uniform(0, 0.1)
            self.beta_1 = np.random.uniform(0, 0.1)
            self.beta_2 = np.random.uniform(0, 0.1)
        else:
            self.alpha_1, self.alpha_2, self.beta_1, self.beta_2 = params

        self.delta = delta  # minimum hiddenstate values before switching

    def hidden_state_derivs(self) -> tuple[float, float]:
        """Return the derivatives of the hidden state."""
        # eqns 3-5
        if self.obs_state == "E":
            dx = -self.alpha_2 * self.x
            dy = self.beta_1 * self.y
        elif self.obs_state == "R":
            dx = self.alpha_1 * self.x
            dy = -self.beta_2 * self.y
        elif self.obs_state == "S":
            dx = self.alpha_1 * self.x
            dy = self.beta_1 * self.y
        return dx, dy
    
    def update_hidden_state(self, stepsize: float = 0.01) -> None:
        """Update hidden state by taking an Euler step."""
        # assert stepsize < self.delta  # ensure we don't take too big a step
        dx, dy = self.hidden_state_derivs()
        self.x += stepsize * dx
        self.y += stepsize * dy
            
    def next_obs_state(self) -> bool:
        """Update observable state. Returns True if the state has changed."""
        # eqn 6
        if self.obs_state in ("R", "S") and self.x >= 1:
            self.obs_state = "E"
            return True
        elif self.obs_state in ("E", "S") and self.x < 1 and self.y >= 1:
            self.obs_state = "R"
            return True
        elif (self.obs_state in ("E", "R") 
              and ((self.x < 1 and self.y <= self.delta) # the leq prevents oob
                   or (self.x <= self.delta and self.y < 1))):
            self.obs_state = "S"
            return True
        elif (self.x > 1 and self.y > 1 and self.obs_state != "S"):
            # tiebreaker. cf 3.0
            self.obs_state = "S"
            return True
        else:
            return False


def simulate_one_cow():
    """Simulate one cow for 1000 steps.
    Make two plots: one of hidden state over time, one of observable state over time.
    """
    cow = Cow()
    steps = 15000
    hidden_states = np.zeros((steps, 2))
    obs_states = np.zeros(steps, dtype=int)
    state_change_idxs = [0]
    for i in range(steps):
        hidden_states[i] = cow.x, cow.y
        obs_states[i] = ["E", "R", "S"].index(cow.obs_state)
        cow.update_hidden_state()
        if cow.next_obs_state():
            print(f"State changed to {cow.obs_state} at step {i}")
            state_change_idxs.append(i)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))  # Increase figure size for more vertical space
    ax1.plot(hidden_states[:, 0], hidden_states[:, 1])
    ax1.scatter(hidden_states[0, 0], hidden_states[0, 1], color='red', label='Initial State')
    for idx in state_change_idxs:
        ax1.annotate(f"{['E', 'R', 'S'][obs_states[idx]]} (s={idx})", (hidden_states[idx, 0], hidden_states[idx, 1]),
                     textcoords="offset points", xytext=(5, 5), ha='center')
    ax1.set_title("Hidden state")
    ax2.plot(obs_states)
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(["E", "R", "S"])
    ax2.set_title("Observable state")
    plt.tight_layout()  # Adjust layout to increase spacing between plots
    plt.show()

if __name__ == "__main__":
    simulate_one_cow()