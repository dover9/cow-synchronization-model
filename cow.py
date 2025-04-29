import numpy as np

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
        if self.x > 1:
            self.x = 1
        if self.y > 1:
            self.y = 1
            
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

    def coupled_hidden_state_derivs(self, neighbor_states: list[str], sigma_x: float, sigma_y: float) -> tuple[float, float]:
        """
        Compute the derivatives of the hidden states (x, y) for a cow interacting with neighbors.

        This method implements the coupled dynamics described in Equation (26) of the paper,
        where the base drift rates for x and y are modified by the influence of neighboring cows.
        Neighbors who are Eating increase the effective growth rate of hunger (x),
        and neighbors who are Resting increase the effective growth rate of tiredness (y).

        Args:
            neighbor_states: List of observable states ("E", "R", "S") of neighboring cows.
            sigma_x: Coupling strength affecting the x-dynamics (hunger).
            sigma_y: Coupling strength affecting the y-dynamics (tiredness).

        Returns:
            dx: Time derivative of the hidden state x (hunger).
            dy: Time derivative of the hidden state y (tiredness).
        
        Notes:
            - If the cow has no neighbors (degree = 0), no coupling is applied.
            - The base drift rates depend on the cow's own observable state:
                * Eating ("E"): alpha = -alpha2, beta = beta1
                * Resting ("R"): alpha = alpha1, beta = -beta2
                * Standing ("S"): alpha = alpha1, beta = beta1
        """
        # Internal rates based on cow's own observable state, from equation 25
        if self.obs_state == "E":
            alpha = -self.alpha_2
            beta = self.beta_1
        elif self.obs_state == "R":
            alpha = self.alpha_1
            beta = -self.beta_2
        elif self.obs_state == "S":
            alpha = self.alpha_1
            beta = self.beta_1
        else:
            raise ValueError(f"Unknown observable state: {self.obs_state}")

        # Neighbor influences
        num_E = sum(1 for state in neighbor_states if state == "E")
        num_R = sum(1 for state in neighbor_states if state == "R")
        degree = len(neighbor_states)

        # Effective rates
        if degree > 0:
            alpha_eff = alpha + (sigma_x / degree) * num_E
            beta_eff = beta + (sigma_y / degree) * num_R
        else:
            alpha_eff = alpha
            beta_eff = beta

        # Coupled derivatives
        dx = alpha_eff * self.x
        dy = beta_eff * self.y

        return dx, dy

