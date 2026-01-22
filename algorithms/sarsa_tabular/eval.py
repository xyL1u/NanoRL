from dataclasses import dataclass
import csv

import numpy as np

from .env_utils import EnvConfig, make_env, reset_env, step_env
from .model import QTable
from .agent import AgentConfig, SarsaAgent
from .train import ensure_output_dirs

@dataclass
class EvalConfig:
    env_id: str
    seed: int
    num_eval_episodes: int = 20
    max_episode_steps: int | None = None
    algorithm_name: str = 'sarsa_tabular'
    render_mode: str | None=None

def evaluate(cfg: EvalConfig) -> dict:
    """
    Evaluate a trained agent using greedy policy (epsilon = 0).

    Args:
        cfg (EvalConfig): Evaluation configuration.

    Returns:
        dict: Evaluation summary, e.g.
            {
              "mean_return": float,
              "std_return": float,
              "mean_length": float,
              "ckpt_path": str
            }
    """
    rng = np.random.default_rng(cfg.seed)
    out = ensure_output_dirs(cfg.algorithm_name)

    env_cfg = EnvConfig(
        env_id=cfg.env_id,
        seed=cfg.seed,
        max_episode_steps=cfg.max_episode_steps,
        render_mode=cfg.render_mode,
        is_eval=True,
    )
    env = make_env(env_cfg)

    # --- Load Q table ---
    ckpt_path = out['ckpt_dir'] / 'q_table.npy'
    qtable = QTable.load(str(ckpt_path))

    n_actions = env.action_space.n
    agent = SarsaAgent(
        AgentConfig(n_actions=n_actions, gamma=0.0, alpha=0.0, epsilon=0.0),
        qtable=qtable,
        rng=rng,
    )

    returns: list[float] = []
    lengths: list[int] = []

    for ep_idx in range(cfg.num_eval_episodes):
        obs = reset_env(env, seed=cfg.seed + ep_idx)
        episode_return, episode_length = 0.0, 0

        while True:
            action = agent.select_action(obs, epsilon=0.0)
            obs_next, r, done, info = step_env(env, action)
            episode_return += r
            episode_length += 1
            obs = obs_next
            if done:
                break
        
        returns.append(episode_return)
        lengths.append(episode_length)

    mean_return = float(np.mean(returns)) if returns else 0.0
    std_return = float(np.std(returns)) if returns else 0.0
    mean_length = float(np.mean(lengths)) if lengths else 0.0

    eval_path = out['result_dir'] / 'eval.csv'
    with open(eval_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'return', 'length'])
        for idx, (ret, length) in enumerate(zip(returns, lengths)):
            writer.writerow([idx, ret, length])
    
    return {
        'mean_return': mean_return,
        'std_return': std_return,
        'mean_length': mean_length,
        'ckpt_path': str(ckpt_path),
    }

if __name__ == '__main__':
    cfg = EvalConfig(
        env_id='CliffWalking-v0',
        seed=43,
        num_eval_episodes=50,
        max_episode_steps=None,
        algorithm_name='sarsa_tabular',
        render_mode=None,
    )

    summary = evaluate(cfg)

    print("=== EVAL DONE ===")
    print(f"Mean return:  {summary['mean_return']:.3f}")
    print(f"Std return:   {summary['std_return']:.3f}")
    print(f"Mean length:  {summary['mean_length']:.2f}")
    print(f"Checkpoint:   {summary['ckpt_path']}")