from dataclasses import dataclass
import gymnasium as gym

@dataclass
class EnvConfig:
    
    env_id: str
    seed: int
    is_eval: bool = False
    max_episode_steps : int | None = None
    render_mode: str | None = None

def make_env(cfg: EnvConfig):
    
    if cfg.max_episode_steps is not None:
        env = gym.make(cfg.env_id, 
                        max_episode_steps=cfg.max_episode_steps, 
                        render_mode = cfg.render_mode, 
        )
    else:
        env = gym.make(cfg.env_id, 
                       render_mode = cfg.render_mode, 
        )
    
    spaces = gym.spaces
    assert isinstance(env.action_space, spaces.Discrete), f"Tabular SARSA requires a Discrete action space, not {env.action_space}."
    assert isinstance(env.observation_space, spaces.Discrete), f"Tabular SARSA requires a Discrete observation space, not {env.observation_space}."

    env.action_space.seed(cfg.seed)

    return env

def reset_env(env, seed: int | None = None):
    
    if seed is not None:
        out = env.reset(seed)
    else:
        out = env.reset()

    obs = out[0]

    return obs

def step_env(env, action):
    
    obs_next, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    return obs_next, reward, done, info