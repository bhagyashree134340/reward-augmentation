import numpy as np
import logging
import gym as gym_
import gymnasium as gym
import gymnasium_robotics
from gymnasium.wrappers import TimeLimit
import minigrid

gym.register_envs(gymnasium_robotics)
gym.register_envs(minigrid)

log = logging.getLogger(__name__)


def make_env(env_name: str, render_mode: str = None, video_dir: str = None, max_episode_steps: int = None, **kwargs):

    if env_name.startswith("Fetch"):
        env = FetchEnvWrapper(gym.make(env_name, render_mode=render_mode, max_episode_steps=max_episode_steps, **kwargs))
    elif env_name.startswith("MiniGrid"):
        env = gym.make(env_name, render_mode=render_mode, **kwargs)
        if max_episode_steps is not None:
            env = TimeLimit(env, max_episode_steps=max_episode_steps)
        env = MiniGridImageObsWrapper(env)
    else:
        env = gym.make(env_name, render_mode=render_mode, max_episode_steps=max_episode_steps, **kwargs)

    return env


class MiniGridImageObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # convert (7, 7, 3) to (3, 7, 7)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(3, 7, 7), dtype=np.uint8
        )

    def observation(self, obs):
        img = obs["image"].transpose(2, 0, 1)  # (3, 7, 7)
        return img


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
