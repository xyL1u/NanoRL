from dataclasses import dataclass
import os
from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt

from .model import QTable, QTableConfig
from .agent import SarsaAgent, AgentConfig
from .env_utils import EnvConfig, make_env, reset_env, step_env

@dataclass
class TrainConfig:
    env_id: str
    seed: int
    num_episodes: int
    max_episodes_step: int | None = None

    # SARSA hyperparameters
    gamma: float = 0.99
    alpha: float = 0.1
    epsilon: float = 0.1

    # Output path
    algorithm_name: str = 'sarsa_tabular'

def ensure_output_dirs(algorithm_name: str) -> dict:
    """
    Create output directories if they do not exist.

    Args:
        algo_name (str): Algorithm folder name under outputs/.

    Returns:
        dict: A dictionary with resolved paths:
            {
              "base": Path,
              "checkpoints": Path,
              "results": Path
            }
    """
    base = Path('outputs') / algorithm_name
    ckpt_dir = base / 'checkpoints'
    result_dir = base / 'results'

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    return {
        'base': base,
        'ckpt_dir': ckpt_dir,
        'result_dir': result_dir,
    }

def train(cfg: TrainConfig) -> dict:
    """
    Run Tabular SARSA training.

    Data flow per episode (SARSA):
        s = reset()
        a = select_action(s)
        while not done:
            s_next, r, done, info = step(a)
            a_next = select_action(s_next)   # on-policy
            learn(s, a, r, s_next, a_next, done)
            s, a = s_next, a_next

    Args:
        cfg (TrainConfig): Training configuration.

    Returns:
        dict: Training summary:
            {
              "episode_returns": list[float],
              "episode_lengths": list[int],
              "qtable_path": str
            }
    """
    # Random generator for reproducibility (agent-side randomness)
    rng = np.random.default_rng(cfg.seed)

    # Outputs
    out = ensure_output_dirs(cfg.algorithm_name)

    # Env
    env_cfg = EnvConfig(
        env_id=cfg.env_id,
        seed=cfg.seed,
        max_episodes_steps=cfg.max_episodes_step,
        render_mode=None,
        is_eval=False,
    )
    env = make_env(env_cfg)

    # QTable dimensions from env (Discrete spaces)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    qtable = QTable(QTableConfig(n_states=n_states, n_actions=n_actions))
    agent = SarsaAgent(
        AgentConfig(n_actions=n_actions, gamma=cfg.gamma, alpha=cfg.alpha, epsilon=cfg.epsilon), 
        qtable=qtable, 
        rng=rng,
    )

    episode_returns: list[float] = []
    episode_lengths: list[int] = []

    for i in range(cfg.num_episodes):
        episode_return = 0.0
        episode_length = 0
        obs = reset_env(env, seed=cfg.seed+i)
        action = agent.select_action(s=obs)
        
        while True:
            obs_next, r, done, info = step_env(env, action)
            episode_return += r
            episode_length += 1
            action_next = agent.select_action(obs_next)
            agent.learn(s=obs, a=action, r=r, s_next=obs_next, a_next=action_next, done=done)
            obs = obs_next
            action = action_next
            if done:
                break
        
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
    
    csv_paths = save_results_csv(out['result_dir'], episode_returns, episode_lengths)
    returns_fig_path = returns_plot(out['result_dir'], episode_returns)
    qtable_path = out['ckpt_dir'] / 'q_table.npy'
    qtable.save(str(qtable_path))

    return {
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
        "qtable_path": str(qtable_path),
        "figure_path": returns_fig_path,
        "rewards_csv": csv_paths['rewards_csv'],
        "lengths_csv": csv_paths['lengths_csv'],
    }

def returns_plot(results_dir, episode_returns):
    """
    Plot and save the training curve of episode returns.

    Args:
        results_dir: Directory path where the figure will be saved.
        episode_returns (list[float]): Total return for each episode.

    Returns:
        str: The saved figure path as a string.
    """
    plt.figure(figsize=(5, 3))
    episodes = list(range(len(episode_returns)))
    plt.plot(episodes, episode_returns, label='returns')
    plt.xlabel('Episode')
    plt.ylabel('Total returns')
    plt.title('Return Curve')
    plt.legend()
    plot_path = results_dir / 'return_curve.png'
    plt.savefig(str(plot_path))
    plt.close()

    return str(plot_path)

def save_results_csv(results_dir, episode_returns, episode_lengths):
    """
    Save per-episode training metrics to CSV files.

    Args:
        results_dir: Directory path where CSV files will be saved.
        episode_returns (list[float]): Total return per episode.
        episode_lengths (list[int]): Episode length (steps) per episode.

    Returns:
        dict: Paths of saved CSV files, e.g.
            {
              "rewards_csv": str,
              "lengths_csv": str
            }
    """
    rewards_path = results_dir / 'rewards.csv'
    lengths_path = results_dir / 'lengths.csv'

    with open(rewards_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'return'])
        for i, ret in enumerate(episode_returns):
            writer.writerow([i, ret])

    with open(lengths_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'length'])
        for i, length in enumerate(episode_lengths):
            writer.writerow([i, length])

    return {
        'rewards_csv': str(rewards_path),
        'lengths_csv': str(lengths_path),
    }