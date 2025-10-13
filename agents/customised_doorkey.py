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
        use_env_coords: bool = True,
        empty_cells: Optional[Iterable[Tuple[int, int]]] = None,  
    ):
        super().__init__(env)
        self.wall_cells = list(wall_cells or [])
        self.goal_cell = goal_cell
        self.use_env_coords = use_env_coords
        self.empty_cells = list(empty_cells or [])  

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        self._apply_patch()

        base_env = self.env.unwrapped
        if hasattr(base_env, "gen_obs"):
            obs = base_env.gen_obs()
        elif hasattr(base_env, "render_obs"):
            obs = base_env.render_obs()
        elif hasattr(base_env, "render"):
            obs = base_env.render()
        else:
            raise RuntimeError("Cannot re-generate observation after patch.")

        return obs, info

    def _to_env(self, x, y):
        return (x, y) if self.use_env_coords else (x + 1, y + 1)

    def _apply_patch(self):
        grid = self.unwrapped.grid

        for x, y in self.empty_cells:
            ex, ey = self._to_env(x, y)
            grid.set(ex, ey, None)

        for x, y in self.wall_cells:
            ex, ey = self._to_env(x, y)
            grid.set(ex, ey, Wall())


class NoDropWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._keep = [Actions.left, Actions.right, Actions.forward,
                      Actions.pickup, Actions.toggle] #removed the done and drop actions 
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
        key_color: str = "yellow",
        door_color: str = "yellow",
        extra_keys=None,
        extra_doors=None,
        **kwargs,
    ):
        assert size >= 5, "Grid too small"
        super().__init__(size=size, **kwargs)

        mid_x = size // 2
        self._fixed_key_pos   = key_pos or (1, size - 2)
        self._fixed_door_pos  = door_pos or (mid_x, size // 2)
        self._fixed_goal_pos  = goal_pos or (size - 2, size // 2)
        self._fixed_agent_start_pos = agent_start_pos or (1, 1)
        self._fixed_agent_start_dir = agent_start_dir

        self._key_color  = key_color
        self._door_color = door_color
        self._extra_keys  = list(extra_keys or [])
        self._extra_doors = list(extra_doors or [])

    def _gen_grid(self, width: int, height: int):
        from minigrid.core.world_object import Key, Door, Wall, Goal
        from minigrid.core.grid import Grid
        import numpy as np

        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        mid_x = width // 2
        door_x, door_y = self._fixed_door_pos
        key_x,  key_y  = self._fixed_key_pos
        goal_x, goal_y = self._fixed_goal_pos

        # Clear interior
        for x in range(1, width - 1):
            for y in range(1, height - 1):
                self.grid.set(x, y, None)

        # Vertical wall with one door gap
        for y in range(1, height - 1):
            if (mid_x, y) == (door_x, door_y):
                continue
            self.grid.set(mid_x, y, Wall())

        # Primary colored door/key
        self.grid.set(door_x, door_y, Door(self._door_color, is_locked=True))
        self.grid.set(key_x,  key_y,  Key(self._key_color))

        # Goal
        self.grid.set(goal_x, goal_y, Goal())

        # Extras
        for (pos, color) in self._extra_keys:
            x, y = pos
            self.grid.set(x, y, Key(color))
        for (pos, color, locked) in self._extra_doors:
            x, y = pos
            self.grid.set(x, y, Door(color, is_open=False, is_locked=bool(locked)))

        # Agent
        ax, ay = self._fixed_agent_start_pos
        assert self.grid.get(ax, ay) is None
        self.agent_pos = np.array([ax, ay], dtype=int)
        self.agent_dir = self._fixed_agent_start_dir

        self.mission = f"open the {self._door_color} door and get to the goal ({width}x{height} fixed layout)"


import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper


import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper
from typing import Optional


import gymnasium as gym
from typing import Optional, List, Tuple
from minigrid.wrappers import FullyObsWrapper
from minigrid.core.world_object import Door, Key, Wall


def make_fixed_doorkey_env(
    # --- core layout ---
    size: int = 10,
    key_color: str = "yellow",
    key_pos: Optional[Tuple[int, int]] = (1, 8),
    door_color: str = "yellow",
    door_pos: Optional[Tuple[int, int]] = (5, 5),
    goal_pos: Optional[Tuple[int, int]] = (8, 1),
    agent_start_pos: Optional[Tuple[int, int]] = (1, 1),
    agent_start_dir: int = 0,
    wall_cells: Optional[List[Tuple[int, int]]] = ((6, 1), (7, 1)),
    # --- wrappers / runtime ---
    max_episode_steps: int = 200,
    render_mode: Optional[str] = "rgb_array",
    disable_env_checker: bool = True,
    use_fully_obs: bool = True,
    use_no_drop: bool = True,
    # --- extras ---
    extra_keys: Optional[List[Tuple[Tuple[int, int], str]]] = None,
    extra_doors: Optional[List[Tuple[Tuple[int, int], str, bool]]] = None,
    ensure_door_in_wall: bool = False,
    overwrite_existing: bool = True,
    seed: Optional[int] = None,
    env_id: str = "Fixed-DoorKey-v0",
    # --- holes to punch after generation ---
    empty_cells: Optional[List[Tuple[int, int]]] = None,
):
    env = gym.make(
        "Fixed-DoorKey-v0",
        size=size,
        key_pos=key_pos,
        door_pos=door_pos,
        goal_pos=goal_pos,
        agent_start_pos=agent_start_pos,
        agent_start_dir=agent_start_dir,
        key_color=key_color,
        door_color=door_color,
        extra_keys=extra_keys,
        extra_doors=extra_doors,
        disable_env_checker=disable_env_checker,
        max_episode_steps=max_episode_steps,
        render_mode=render_mode,
        highlight=False
    )

    if wall_cells is not None or goal_pos is not None or empty_cells is not None:
        env = PatchGridWrapper(
            env,
            wall_cells=list(wall_cells) if wall_cells is not None else None,
            goal_cell=tuple(goal_pos) if goal_pos is not None else None,
            empty_cells=list(empty_cells) if empty_cells is not None else None,  
        )

    if use_fully_obs:
        env = FullyObsWrapper(env)
    if use_no_drop:
        env = NoDropWrapper(env)

    # Recolor / re-place primary objects post-hoc
    base = env.unwrapped
    grid = base.grid
    def _set_cell(pos: Tuple[int, int], obj):
        x, y = pos
        existing = grid.get(x, y)
        if existing is not None and not overwrite_existing:
            return
        grid.set(x, y, obj)

    if key_pos is not None and key_color:
        _set_cell(tuple(key_pos), Key(key_color))
    if door_pos is not None and door_color:
        _set_cell(tuple(door_pos), Door(color=door_color, is_open=False, is_locked=True))

    if extra_keys:
        for (pos, color) in extra_keys:
            _set_cell(tuple(pos), Key(color))
    if extra_doors:
        for (pos, color, locked) in extra_doors:
            if ensure_door_in_wall:
                cell = grid.get(*pos)
                if not isinstance(cell, Wall):
                    print(f"[make_fixed_doorkey_env] Warning: door at {pos} is not on a Wall cell.")
            _set_cell(tuple(pos), Door(color=color, is_open=False, is_locked=bool(locked)))

    
    if ensure_door_in_wall and door_pos is not None:
        mid_x = size // 2
        if door_pos[0] != mid_x:
            print(f"[make_fixed_doorkey_env] Warning: primary door {door_pos} is not on center wall x={mid_x}.")

    if seed is not None:
        env.reset(seed=seed)

    return env



register(
    id="Fixed-DoorKey-v0",  
    entry_point=__name__ + ":FixedDoorKeyEnv",
)