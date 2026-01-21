from dataclasses import dataclass
import numpy as np

from .model import QTable

@dataclass
class AgentConfig:
    n_actions: int
    gamma: float
    alpha: float
    epsilon: float

class SarsaAgent:
    def __init__(self, cfg: AgentConfig, qtable: QTable, rng: np.random.Generator | None = None):
        self.cfg = cfg
        self.q = qtable
        self.rng = rng if rng is not None else np.random.default_rng()

    def select_action(self, s: int, epsilon: float | None = None) -> int:
        """
        Select an action using epsilon-greedy policy.

        With probability epsilon: choose a random action uniformly from [0, n_actions).
        With probability (1 - epsilon): choose the greedy action argmax_a Q(s, a).

        Args:
            s (int): Discrete state index.
            epsilon (float | None): Exploration rate. If None, uses self.cfg.epsilon.

        Returns:
            int: Selected discrete action index.
        """
        eps = self.cfg.epsilon if epsilon is None else epsilon
        assert 0.0 <= eps <= 1.0, f'epsilon must be in [0,1)'

        u = self.rng.random()

        if u < eps:
            action = self.rng.integers(low=0, high=self.cfg.n_actions)
        else:
            action = self.q.argmax(s)
        
        return int(action)
    
    def learn(self, s: int, a: int, r: float, s_next: int, a_next: int, done: bool):
        """
        Perform one Tabular SARSA update.

        SARSA target:
            if done:
                y = r
            else:
                y = r + gamma * Q(s_next, a_next)

        TD error:
            delta = y - Q(s, a)

        Update:
            Q(s, a) <- Q(s, a) + alpha * delta

        Args:
            s (int): Current discrete state index.
            a (int): Action taken at state s.
            r (float): Reward observed after taking action a.
            s_next (int): Next discrete state index.
            a_next (int): Next action chosen by the same behavior policy (epsilon-greedy).
            done (bool): True if episode ended (terminated or truncated).

        Returns:
            dict: Logging stats (useful for debugging / plotting), e.g.
                {
                    "q_sa": float,
                    "q_snext_anext": float,
                    "td_target": float,
                    "td_error": float,
                    "new_q": float
                }
        """
        gamma = self.cfg.gamma
        alpha = self.cfg.alpha
        r = float(r)

        # Q(s, a)
        q_sa = self.q.get(s, a)
        
        if not done:
            q_snext_anext = self.q.get(s_next, a_next)
        else:
            q_snext_anext = 0.0
        
        td_target = r + gamma * q_snext_anext

        td_error = td_target - q_sa

        new_q = q_sa + alpha * td_error
        self.q.set(s, a, new_q)

        return {
            'q_sa': float(q_sa),
            'td_target': float(td_target),
            'td_error': float(td_error),
            'q_snext_anext': float(q_snext_anext),
            'new_q': float(new_q),
        }