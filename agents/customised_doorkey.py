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

from minigrid.core.world_object import Wall, Goal
import gymnasium as gym

from typing import Optional, Iterable, Tuple
from gymnasium.core import Wrapper
from minigrid.core.world_object import Wall, Goal

class PatchGridWrapper(Wrapper):
    def __init__(
        self,
        env,
        wall_cells: Optional[Iterable[Tuple[int, int]]] = None,
        goal_cell: Optional[Tuple[int, int]] = None,
        use_env_coords: bool = False,
    ):
        super().__init__(env)
        self.wall_cells = list(wall_cells or [])
        self.goal_cell = goal_cell
        self.use_env_coords = use_env_coords

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._apply_patch()
        return obs, info

    def _to_env(self, x, y):
        return (x, y) if self.use_env_coords else (x + 1, y + 1)

    def _apply_patch(self):
        grid = self.unwrapped.grid

        for x, y in self.wall_cells:
            ex, ey = self._to_env(x, y)
            grid.set(ex, ey, Wall())

        if self.goal_cell is not None:
            gx, gy = self.goal_cell
            ex, ey = self._to_env(gx, gy)
            grid.set(ex, ey, Goal())



class NoDropWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._keep = [Actions.left, Actions.right, Actions.forward,
                      Actions.pickup, Actions.toggle, Actions.done]
        self.action_space = spaces.Discrete(len(self._keep))

    def step(self, a):
        a_mapped = self._keep[a]
        return self.env.step(a_mapped)
    

class FixedDoorKeyEnv(DoorKeyEnv):
    metadata = getattr(DoorKeyEnv, "metadata", {})

    def __init__(
        self,
        size: int = 6,
        key_pos=None,
        door_pos=None,
        goal_pos=None,
        agent_start_pos=None,
        agent_start_dir: int = 0,
        **kwargs,
    ):
        assert size >= 5, "Grid too small"
        super().__init__(size=size, **kwargs)

        mid_x = size // 2

        self._fixed_key_pos = key_pos or (1, size - 2)               
        self._fixed_door_pos = door_pos or (mid_x, size // 2)        
        self._fixed_goal_pos = goal_pos or (size - 2, size // 2)     
        self._fixed_agent_start_pos = agent_start_pos or (1, 1)      
        self._fixed_agent_start_dir = agent_start_dir

    def _gen_grid(self, width: int, height: int):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        mid_x = width // 2
        door_x, door_y = self._fixed_door_pos
        key_x, key_y = self._fixed_key_pos
        goal_x, goal_y = self._fixed_goal_pos

        for x in range(1, width - 1):
            for y in range(1, height - 1):
                self.grid.set(x, y, None)

        for y in range(1, height - 1):
            if (mid_x, y) == (door_x, door_y):
                continue
            self.grid.set(mid_x, y, Wall())

        self.grid.set(door_x, door_y, Door("yellow", is_locked=True))

        self.grid.set(key_x, key_y, Key("yellow"))

        self.grid.set(goal_x, goal_y, Goal())

        ax, ay = self._fixed_agent_start_pos
        assert self.grid.get(ax, ay) is None
        self.agent_pos = np.array([ax, ay], dtype=int)
        self.agent_dir = self._fixed_agent_start_dir

        self.mission = f"open the yellow door and get to the goal ({width}x{height} fixed layout)"


register(
    id="Fixed-DoorKey-v0",  
    entry_point=__name__ + ":FixedDoorKeyEnv",
)