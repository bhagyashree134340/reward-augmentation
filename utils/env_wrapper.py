import numpy as np
import logging
import gymnasium as gym
import gymnasium_robotics
from gymnasium.wrappers import TimeLimit

gym.register_envs(gymnasium_robotics)

log = logging.getLogger(__name__)


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
        # self._set_object_position(offset=(0.0, 0.1))
        # self._set_goal_position(offset=(0.1, 0.1, 0.0))
        # obs = self._flatten_obs(self.env._get_obs())
        # return obs, info
        return (self._flatten_obs(obs), info) if self.is_goal_env else (obs, info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.is_goal_env:
            obs = self._flatten_obs(obs)
        return obs, reward, terminated, truncated, info
    
    # def _set_object_position(self, offset=(0.0, 0.0)):
    #     env = self.env.unwrapped
    #     object_xpos = env.initial_gripper_xpos[:2] + np.array(offset)
    #     object_qpos = env.sim.data.get_joint_qpos("object0:joint").copy()
    #     object_qpos[:2] = object_xpos
    #     env.sim.data.set_joint_qpos("object0:joint", object_qpos)
    #     env.sim.forward()

    # def _set_goal_position(self, offset=(0.0, 0.0, 0.0)):
    #     env = self.env.unwrapped
    #     goal = env.initial_gripper_xpos.copy()
    #     goal[2] = env.height_offset
    #     goal += env.target_offset
    #     goal += np.array(offset)
    #     env.goal = goal.copy()
