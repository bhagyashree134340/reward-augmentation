import numpy as np
import torch
import matplotlib.pyplot as plt
import wandb
from minigrid.core.world_object import Door, Key

def _find_first(env, cls):
    """Return (x,y,obj) of the first object of type cls in the grid."""
    for x in range(env.grid.width):
        for y in range(env.grid.height):
            obj = env.grid.get(x, y)
            if isinstance(obj, cls):
                return (x, y, obj)
    return None

def _set_door_state(env, open_: bool):
    """Force the door to be open/closed for rendering (no physics)."""
    loc = _find_first(env, Door)
    if loc is None:
        return
    x, y, door = loc
    # Force state directly (faster/robust for probes)
    door.is_open = open_
    # Keep it unlocked to avoid weird visuals if you step
    door.is_locked = False
    env.grid.set(x, y, door)

def _set_inventory(env, has_key: bool):
    """Show the agent as carrying/not-carrying a key for rendering."""
    if has_key:
        # Use the door color if present; fallback to 'yellow'
        loc = _find_first(env, Door)
        key_color = loc[2].color if loc else "yellow"
        env.carrying = Key(key_color)
        env.carrying.cur_pos = None  # carried
    else:
        env.carrying = None

def _is_walkable(env, x, y):
    """Mask walls; also optional: mask cells outside inner room."""
    obj = env.grid.get(x+1, y+1)  # +1 because of outer wall border
    # treat None (empty) or Goal/Key/door tiles as "valid to visualize"
    return True if obj is None or getattr(obj, "can_overlap", True) else False

def _render_probe_obs(agent, base_env, x, y, dir_idx, has_key, door_open):
    """
    Configure env to the requested slice and return processed obs.
    """
    env = base_env
    env.reset()  # ensure a clean base; your FixedDoorKeyEnv keeps layout constant
    # Make rendering match training resolution
    try:
        env.tile_size = 4  # match your RGBImgObsWrapper(tile_size=4)
    except Exception:
        pass

    # World state slice
    _set_door_state(env, door_open)
    _set_inventory(env, has_key)

    # Place agent (grid has one-cell outer wall, so shift by +1)
    env.agent_pos = (x + 1, y + 1)
    env.agent_dir = dir_idx  # 0:N,1:E,2:S,3:W

    # Render exactly the same input domain you trained on
    obs_raw = env.render()   # rgb array
    obs = agent.process_obs(obs_raw)  # -> torch-ready tensor or np (your code)
    return obs


def log_small_multiples_heatmaps(agent, step, grid_h=8, grid_w=8, mask_walls=True):
    """
    Creates 16 heatmaps (dir x has_key x door_open) of intrinsic bonus over (x,y).
    Logs a single 4x4 grid image to WandB.
    """
    base_env = agent.eval_env.unwrapped  # or a separate fixed env instance
    dirs = [0, 1, 2, 3]                 # N,E,S,W
    has_keys = [False, True]
    doors_open = [False, True]

    panels = []
    titles = []

    # Precompute mask (optional) so walls show as NaN
    wall_mask = np.ones((grid_h, grid_w), dtype=bool)
    if mask_walls:
        for y in range(grid_h):
            for x in range(grid_w):
                wall_mask[y, x] = _is_walkable(base_env, x, y)

    for d in dirs:
        for hk in has_keys:
            for do in doors_open:
                H = np.full((grid_h, grid_w), np.nan, dtype=np.float32)
                for y in range(grid_h):
                    for x in range(grid_w):
                        if mask_walls and not wall_mask[y, x]:
                            continue
                        obs = _render_probe_obs(agent, base_env, x, y, d, hk, do)
                        # to torch tensor (match your compute function’s expectations)
                        if isinstance(obs, np.ndarray):
                            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(agent.device)
                        else:
                            # already torch; ensure batch dim + device
                            obs_t = obs.unsqueeze(0).to(agent.device)

                        with torch.no_grad():
                            bonus = agent.compute_intrinsic_bonus(obs_t)  # shape [1] or [1,1]
                        H[y, x] = float(bonus.squeeze().cpu().item())

                panels.append(H)
                titles.append(f"dir={d} | key={int(hk)} | door_open={int(do)}")

    # Plot 4x4
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    vmax = np.nanpercentile([p for P in panels for p in P.ravel() if not np.isnan(p)], 95)
    vmin = 0.0
    for ax, H, title in zip(axes.ravel(), panels, titles):
        im = ax.imshow(H, origin="upper", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    # single colorbar for all
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    cbar.set_label("intrinsic bonus", rotation=90)
    fig.suptitle(f"CFN/RND intrinsic bonus small-multiples @ step {step}", fontsize=14)
    fig.tight_layout()

    wandb.log({"intrinsic/small_multiples": wandb.Image(fig)}, step=step)
    plt.close(fig)
