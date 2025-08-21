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
    16 heatmaps (dir x has_key x door_open) of intrinsic bonus over (x,y).
    Logs a single 4x4 grid image to WandB.
    """
    base_env = agent.eval_env.unwrapped
    dirs = [0, 1, 2, 3]                 # N,E,S,W
    has_keys = [False, True]
    doors_open = [False, True]

    # --- ensure deterministic eval for the probe ---
    try:
        agent.rnd_predictor.eval()      # or agent.cfn.eval(), depending on your class
    except Exception:
        pass

    panels, titles = [], []

    # Precompute wall mask once (just to hide outer walls in plots)
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

                        # Render probe obs exactly like training (but silence debug prints)
                        obs = _render_probe_obs(agent, base_env, x, y, d, hk, do)
                        if isinstance(obs, np.ndarray):
                            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(agent.device)
                        else:
                            obs_t = obs.float().unsqueeze(0).to(agent.device)

                        with torch.inference_mode():
                            b = agent._bonus_from_obs_tensor(obs_t)
                            if isinstance(b, torch.Tensor):
                                H[y, x] = b.detach().flatten()[0].item()
                            try:
                                H[y, x] = float(b)
                            except Exception:
                                H[y, x] = float(np.asarray(b).flatten()[0])

                panels.append(H)
                titles.append(f"dir={d} | key={int(hk)} | door_open={int(do)}")

    # --- plotting (use constrained layout; no tight_layout) ---
    fig, axes = plt.subplots(4, 4, figsize=(16, 16), constrained_layout=True)

    # robust color scaling across all panels (95th percentile)
    finite_vals = np.concatenate([P[np.isfinite(P)].ravel() for P in panels])
    vmax = np.percentile(finite_vals, 95) if finite_vals.size else 1.0
    vmin = 0.0

    last_im = None
    for ax, H, title in zip(axes.ravel(), panels, titles):
        last_im = ax.imshow(H, origin="upper", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

    cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.8)
    cbar.set_label("intrinsic bonus", rotation=90)
    fig.suptitle(f"CFN/RND intrinsic bonus small-multiples @ step {step}", fontsize=14)

    wandb.log({"intrinsic/small_multiples": wandb.Image(fig)}, step=step)
    plt.close(fig)
