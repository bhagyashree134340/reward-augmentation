import numpy as np
from cpprb import ReplayBuffer, HindsightReplayBuffer
from gymnasium.spaces import Box, Dict as SpaceDict


def make_replay_buffer(env, buffer_size=1_000_000, use_her=True, max_episode_len=100):
    obs_space = env.observation_space
    act_space = env.action_space

    # HER only makes sense if the env uses a Dict observation space with desired_goal
    is_goal_env = isinstance(obs_space, SpaceDict) and "desired_goal" in obs_space.spaces
    use_her = use_her and is_goal_env

    if use_her:
        print("Using HER-enabled replay buffer.")

        obs_shape = obs_space["observation"].shape
        goal_shape = obs_space["desired_goal"].shape
        act_shape = act_space.shape

        def reward_func(achieved_goal, desired_goal, _info):
            return (np.linalg.norm(achieved_goal - desired_goal, axis=-1) < 0.05).astype(np.float32)

        env_dict = {
            "obs": {"shape": obs_shape},
            "act": {"shape": act_shape},
            "next_obs": {"shape": obs_shape},
            "done": {},
            "rew": {},
            "goal": {"shape": goal_shape},
            "next_goal": {"shape": goal_shape},
            "achieved_goal": {"shape": goal_shape},
            "next_achieved_goal": {"shape": goal_shape},
        }

        return HindsightReplayBuffer(
            size=buffer_size,
            env_dict=env_dict,
            max_episode_len=max_episode_len,
            reward_func=reward_func,
        )

    else:
        return ReplayBuffer(
            size=buffer_size,
            env_dict={
                "obs": {"shape": env.observation_space.shape[0]},
                "act": {"shape": env.action_space.shape[0]},
                "ext_rew": {},
                "int_rew": {},
                "next_obs": {"shape": env.observation_space.shape[0]},
                "done": {}
            }
        )
