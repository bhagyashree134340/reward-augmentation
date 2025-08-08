# fixed_doorkey.py
from gymnasium.envs.registration import register
from minigrid.envs.doorkey import DoorKeyEnv
from minigrid.core.world_object import Key, Door, Wall, Goal
from minigrid.core.grid import Grid
import gymnasium as gym
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import Wrapper
from minigrid.core.actions import Actions

class NoDropWrapper(Wrapper):
    """
    Remove the Drop action from the action space.
    Maps a reduced Discrete to the original actions without Actions.drop.
    """
    def __init__(self, env):
        super().__init__(env)
        # original order: use the enum so we don't guess indices
        self._keep = [Actions.left, Actions.right, Actions.forward,
                      Actions.pickup, Actions.toggle, Actions.done]
        self.action_space = spaces.Discrete(len(self._keep))

    def step(self, a):
        a_mapped = self._keep[a]
        return self.env.step(a_mapped)
    

class FixedDoorKeyEnv(DoorKeyEnv):
    """
    DoorKey-6x6 with fixed positions for the key and the door.

    Args:
        size (int): must be 6 for the classic DoorKey-6x6 layout.
        key_pos (tuple[int,int] | None): (x, y) fßßßor the yellow key (must be a free floor tile, not a wall).
        door_pos (tuple[int,int] | None): (x, y) for the yellow door (must be on the internal middle wall x = width // 2).
        agent_start_pos (tuple[int,int] | None): fixed agent start (must be a free floor tile). If None, keep default random.
        agent_start_dir (int): 0:right, 1:down, 2:left, 3:up.
        kwargs: forwarded to DoorKeyEnv (reward, etc.)
    """
    metadata = getattr(DoorKeyEnv, "metadata", {})
    from typing import Optional, Tuple

    def __init__(
        self,
        size: int = 6,
        key_pos: Optional[Tuple[int, int]] = (1, 4),
        door_pos: Optional[Tuple[int, int]] = (3, 3),
        goal_pos: Optional[Tuple[int, int]] = (4, 4),
        agent_start_pos: Optional[Tuple[int, int]] = None,
        agent_start_dir: int = 0,
        **kwargs,
    ):
        assert size == 6, "This variant uses the 6x6 DoorKey layout. Change checks if you want another size."
        super().__init__(size=size, **kwargs)
        self._fixed_key_pos = key_pos
        self._fixed_door_pos = door_pos
        self._fixed_goal_pos = goal_pos
        self._fixed_agent_start_pos = agent_start_pos
        self._fixed_agent_start_dir = agent_start_dir

    def _gen_grid(self, width: int, height: int):
        # Build empty grid with surrounding walls
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # === Internal layout ===
        mid_x = width // 2
        door_x, door_y = self._fixed_door_pos
        key_x, key_y = self._fixed_key_pos
        goal_x, goal_y = getattr(self, "_fixed_goal_pos", (width - 2, height - 2))

        # Clear internal area
        for x in range(1, width - 1):
            for y in range(1, height - 1):
                self.grid.set(x, y, None)

        # Rebuild middle wall, skip door position
        for y in range(1, height - 1):
            if y == door_y:
                continue
            self.grid.set(mid_x, y, Wall())

        # Place fixed door
        self.grid.set(door_x, door_y, Door("yellow", is_locked=True))

        # Place fixed key
        self.grid.set(key_x, key_y, Key("yellow"))

        # Place fixed goal (optional)
        self.grid.set(goal_x, goal_y, Goal())

        # Place agent
        if self._fixed_agent_start_pos is not None:
            ax, ay = self._fixed_agent_start_pos
            assert self.grid.get(ax, ay) is None
            self.agent_pos = np.array([ax, ay], dtype=int)
            self.agent_dir = self._fixed_agent_start_dir
        else:
            raise RuntimeError("Fixed agent start position is required")

        self.mission = "open the yellow door and get to the goal (fixed)"


# Register a Gymnasium ID for convenience
register(
    id="Fixed-DoorKey-6x6-v0",
    entry_point=__name__ + ":FixedDoorKeyEnv",
)



# import os
# import imageio
# import numpy as np
# import gymnasium as gym
# from minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper, ImgObsWrapper

# # Output folder
# os.makedirs("gifs", exist_ok=True)

# # Make the env with your fixed positions
# env = gym.make(
#     "Fixed-DoorKey-6x6-v0",
#     disable_env_checker=True,
#     render_mode="rgb_array",
#     key_pos=(1, 4),
#     door_pos=(3, 3),
#     agent_start_pos=(1, 1),   # optional but avoids assertions
# )
# env = FullyObsWrapper(env)
# env = RGBImgObsWrapper(env, tile_size=8)
# env = ImgObsWrapper(env)

# # Generate N episodes as GIFs
# n_gifs = 4
# max_steps = 30  # enough to move around a bit to see positions

# for ep in range(n_gifs):
#     frames = []
#     obs, info = env.reset(seed=ep)  # different seed → layout should be same
#     frames.append(env.render())     # capture first frame

#     done = False
#     step = 0
#     while not done and step < max_steps:
#         action = env.action_space.sample()
#         obs, reward, terminated, truncated, info = env.step(action)
#         frames.append(env.render())
#         step += 1
#         done = terminated or truncated

#     # Save to GIF
#     gif_path = f"gifs/fixed_doorkey_ep{ep}.gif"
#     imageio.mimsave(gif_path, frames, duration=0.2)
#     print(f"Saved {gif_path}")

# env.close()