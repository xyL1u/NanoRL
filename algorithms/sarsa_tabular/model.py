from dataclasses import dataclass
import numpy as np

@dataclass
class QTableConfig:
    """
    Configuration class for the Q-table.

    Attributes
    ----------
    n_states : int
        Total number of discrete states in the environment.
    n_actions : int
        Total number of discrete actions available in each state.
    init_value : float, optional
        Initial value used to fill the Q-table (default is 0.0).
        This is usually set to 0.0 for tabular RL.
    dtype : type, optional
        Data type of the Q-table values (default is np.float32).
        Using float32 saves memory and is sufficient for RL.
    """
    n_states: int
    n_actions: int
    init_value: float = 0.0
    dtype: type = np.float32

class QTable:
    """
    A simple tabular Q-function implementation for discrete
    state and action spaces.

    The Q-table stores Q(s, a) values in a 2D NumPy array
    of shape (n_states, n_actions).
    """
    
    def __init__(self, cfg: QTableConfig):
        """
        Initialize the Q-table using the provided configuration.

        Parameters
        ----------
        cfg : QTableConfig
            Configuration object specifying table size,
            initialization value, and data type.
        """
        self.cfg = cfg
        self.q = np.full((cfg.n_states, cfg.n_actions), cfg.init_value, dtype=cfg.dtype)

    def get(self, s: int, a: int) -> float:
        """
        Get the Q-value for a specific state-action pair.

        Parameters
        ----------
        s : int
            State index.
        a : int
            Action index.

        Returns
        -------
        float
            Q(s, a) value as a Python float.
        """
        return float(self.q[s, a])
    
    def set(self, s: int, a: int, value: float) -> None:
        """
        Set the Q-value for a specific state-action pair.

        Parameters
        ----------
        s : int
            State index.
        a : int
            Action index.
        value : float
            New Q-value to assign.
        """
        self.q[s, a] = value

    def row(self, s: int):
        return self.q[s, :]
    
    def argmax(self, s: int) -> int:
        return int(np.argmax(self.row(s)))
    
    def save(self, path: str) -> None:
        np.save(path, self.q)

    @classmethod
    def load(cls, path: str):
        """
        Load a Q-table from a NumPy .npy file and reconstruct
        a QTable object.

        Parameters
        ----------
        path : str
            Path to the saved .npy file.

        Returns
        -------
        QTable
            A QTable instance initialized with the loaded data.
        """
        q = np.load(path)
        num_states, num_actions = q.shape
        
        # Recreate the configuration using loaded metadata
        cfg = QTableConfig(
            n_states=num_states, 
            n_actions=num_actions, 
            init_value=0.0,
            dtype=q.dtype.type,
        )
        
        # Create a new QTable instance and overwrite its Q-table
        obj = cls(cfg)
        obj.q = q
        return obj
    