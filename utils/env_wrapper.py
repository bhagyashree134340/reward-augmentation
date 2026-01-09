import numpy as np
import logging
import gymnasium as gym
import gymnasium_robotics
from gymnasium.wrappers import TimeLimit

gym.register_envs(gymnasium_robotics)

log = logging.getLogger(__name__)

import gymnasium as gym
import numpy as np

class ActionConfusionWrapper(gym.ActionWrapper):
    """
    Replaces the intended discrete action 'a' with a sampled action 'a_exec'
    according to a confusion matrix M[a, a_exec].
    If 'uniform_flip_p' is given, builds M such that with prob p we flip
    uniformly to any *other* action, else execute as intended.
    """
    def __init__(self, env, confusion_matrix=None, uniform_flip_p=None, seed=None):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Discrete), "Requires discrete actions."
        self.nA = env.action_space.n
        if confusion_matrix is not None:
            M = np.asarray(confusion_matrix, dtype=np.float64)
            assert M.shape == (self.nA, self.nA)
            # normalize rows just in case
            M = M / M.sum(axis=1, keepdims=True)
            self.M = M
        else:
            assert uniform_flip_p is not None and 0.0 <= uniform_flip_p <= 1.0
            p = uniform_flip_p
            M = np.full((self.nA, self.nA), p / (self.nA - 1), dtype=np.float64)
            np.fill_diagonal(M, 1.0 - p)
            self.M = M
        self.rng = np.random.default_rng(seed)

    def action(self, a_intended):
        # sample executed action given intended action
        probs = self.M[a_intended]
        a_exec = self.rng.choice(self.nA, p=probs)
        # store for optional logging (debug)
        self.last_intended = a_intended
        self.last_executed = a_exec
        return a_exec



def make_env(env_name: str, render_mode: str = None, video_dir: str = None, max_episode_steps: int = None, **kwargs):

    if env_name.startswith("Fetch"):
        env = FetchEnvWrapper(gym.make(env_name, render_mode=render_mode, max_episode_steps=max_episode_steps, **kwargs))
    else:
        env = gym.make(env_name, render_mode=render_mode, max_episode_steps=max_episode_steps, **kwargs)

    return env


class FetchEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.is_goal_env = isinstance(env.observation_space, gym.spaces.Dict)

        if self.is_goal_env:
            self.observation_space = self._get_flat_observation_space()

    def _flatten_obs(self, obs):
        return np.concatenate([
            np.asarray(obs['observation'], dtype=np.float32),
            np.asarray(obs['desired_goal'], dtype=np.float32)
        ], axis=0)

    def _get_flat_observation_space(self):
        obs_space = self.env.observation_space['observation']
        goal_space = self.env.observation_space['desired_goal']

        low = np.concatenate([obs_space.low, goal_space.low], axis=0, dtype=np.float32)
        high = np.concatenate([obs_space.high, goal_space.high], axis=0, dtype=np.float32)

        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return (self._flatten_obs(obs), info) if self.is_goal_env else (obs, info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.is_goal_env:
            obs = self._flatten_obs(obs)
        return obs, reward, terminated, truncated, info
