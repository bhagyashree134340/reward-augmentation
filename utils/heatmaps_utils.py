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
    env.agent_dir = dir_idx  # 0-U,1-R,2:D,3:L

    # Render exactly the same input domain you trained on
    obs_raw = env.render()   # rgb array
    obs = agent.process_obs(obs_raw)  # -> torch-ready tensor or np (your code)
    return obs


def log_small_multiples_heatmaps(agent, step, grid_h=8, grid_w=8, mask_walls=True, method_name="cfn"):
    """
    16 heatmaps (dir x has_key x door_open) of intrinsic bonus over (x,y).
    Logs a single 4x4 grid image to WandB.
    """
    base_env = agent.eval_env.unwrapped
    dirs = [0, 1, 2, 3]                 # N,E,S,W up right down left
    has_keys = [False, True]
    doors_open = [False, True]

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
                            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(agent.device) / 255.0
                        else:
                            obs_t = obs.float().unsqueeze(0).to(agent.device) / 255.0

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
    fig.suptitle(f"CFN intrinsic bonus @ step {step}", fontsize=14)

    wandb.log({"intrinsic/small_multiples": wandb.Image(fig)}, step=step)
    plt.close(fig)

def plot_cfn_difficulty_panels(agent, step, show_counts=True, add_scatter=True,
                               figsize=(18, 6), percentiles=(1, 99), font=12):
    import math, numpy as np, matplotlib.pyplot as plt, torch
    import matplotlib.patheffects as pe
    from minigrid.core.world_object import Wall, Door, Key

    base = agent.eval_env.unwrapped
    try: base.tile_size = 4
    except Exception: pass

    H = base.grid.height - 2
    W = base.grid.width  - 2

    door_pos = key_pos = goal_pos = None
    for y in range(1, base.grid.height - 1):
        for x in range(1, base.grid.width  - 1):
            obj = base.grid.get(x, y)
            if isinstance(obj, Door): door_pos = (x-1, y-1)
            elif isinstance(obj, Key): key_pos  = (x-1, y-1)
            elif getattr(obj, "type", None) == "goal": goal_pos = (x-1, y-1)

    def set_world_state(door_open=None, has_key=False):
        base.reset()
        base.carrying = None
        if has_key:
            color = 'yellow'
            if door_pos is not None:
                d = base.grid.get(door_pos[0]+1, door_pos[1]+1)
                if isinstance(d, Door) and hasattr(d, "color"): color = d.color
            base.carrying = Key(color=color)
        if door_pos is not None and door_open is not None:
            d = base.grid.get(door_pos[0]+1, door_pos[1]+1)
            if isinstance(d, Door): d.is_open = bool(door_open)

    def reachability(door_open=None, has_key=False):
        """Only used for the optional scatter."""
        from collections import deque
        occ = np.zeros((H, W), dtype=bool)
        for yy in range(H):
            for xx in range(W):
                obj = base.grid.get(xx+1, yy+1)
                if isinstance(obj, Wall): occ[yy, xx] = True
                elif isinstance(obj, Door): occ[yy, xx] = not (door_open or has_key)
        sx, sy = getattr(base, "agent_pos", (1, 1))
        sx, sy = sx-1, sy-1
        sx = np.clip(sx, 0, W-1); sy = np.clip(sy, 0, H-1)
        reach = np.zeros_like(occ, dtype=bool)
        dist  = np.full_like(occ, np.inf, dtype=float)
        if not occ[sy, sx]:
            q = deque([(sx, sy)]); reach[sy, sx] = True; dist[sy, sx] = 0
            while q:
                x, y = q.popleft()
                for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < W and 0 <= ny < H and not occ[ny, nx] and not reach[ny, nx]:
                        reach[ny, nx] = True; dist[ny, nx] = dist[y, x] + 1; q.append((nx, ny))
        return reach, dist

    @torch.no_grad()
    def bonus_map(door_open=None, has_key=False):
        set_world_state(door_open, has_key)
        M = np.full((H, W), np.nan, dtype=np.float32)
        for y in range(H):
            for x in range(W):
                if isinstance(base.grid.get(x+1, y+1), Wall): continue
                vals = []
                for d in (0,1,2,3):
                    base.agent_pos = (x+1, y+1); base.agent_dir = d
                    base.step(base.actions.toggle)  # refresh visuals
                    obs = agent.process_obs(base.render())
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0) / 255.0
                    pred = agent.cfn(obs_t, update_prior_stats=False)
                    vals.append((pred.norm(p=2, dim=1) / math.sqrt(agent.coin_flip_dim)).item())
                M[y, x] = float(np.mean(vals))
        return M

    panels = [("Closed, no key", dict(door_open=False, has_key=False)),
              ("Closed, has key", dict(door_open=False, has_key=True)),
              ("Door open",       dict(door_open=True,  has_key=False))]

    maps = [bonus_map(**cond) for _, cond in panels]

    # shared color scale
    all_vals = np.concatenate([m[~np.isnan(m)] for m in maps if np.any(~np.isnan(m))]) if maps else np.array([])
    if all_vals.size:
        if percentiles is None:
            vmin, vmax = float(all_vals.min()), float(all_vals.max())
        else:
            lo, hi = percentiles
            vmin, vmax = np.percentile(all_vals, lo), np.percentile(all_vals, hi)
    else:
        vmin = vmax = None

    # layout: 3 panels + dedicated colorbar axis
    fig = plt.figure(figsize=figsize, constrained_layout=True, dpi=120)
    gs  = fig.add_gridspec(nrows=1, ncols=4, width_ratios=[1,1,1,0.035])
    axs = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])

    txt_pe = [pe.withStroke(linewidth=2, foreground="black")]
    last_im = None
    for ax, (title, _), M in zip(axs, panels, maps):
        show = np.clip(M, vmin, vmax) if vmin is not None else M
        last_im = ax.imshow(show, cmap="viridis", vmin=vmin, vmax=vmax,
                            interpolation="nearest", origin="upper")
        ax.set_title(title, fontsize=font+2, pad=8)
        ax.set_xticks(range(W)); ax.set_yticks(range(H))
        ax.tick_params(labelsize=font-2)
        # markers
        if door_pos: ax.add_patch(plt.Rectangle((door_pos[0]-0.5, door_pos[1]-0.5), 1, 1, fill=False, lw=2))
        if door_pos: ax.text(*door_pos, "D", ha="center", va="center", fontsize=font, weight="bold", color="white", path_effects=txt_pe)
        if key_pos:  ax.text(*key_pos,  "K", ha="center", va="center", fontsize=font, weight="bold", color="white", path_effects=txt_pe)
        if goal_pos: ax.text(*goal_pos, "G", ha="center", va="center", fontsize=font, weight="bold", color="white", path_effects=txt_pe)
        # counts
        if show_counts and hasattr(agent, "visit_counts") and agent.visit_counts.shape == (H, W):
            for yy in range(H):
                for xx in range(W):
                    ax.text(xx, yy, str(int(agent.visit_counts[yy, xx])),
                            ha="center", va="center", fontsize=font-3, color="white", path_effects=txt_pe)

    if last_im is not None:
        cb = fig.colorbar(last_im, cax=cax)
        cb.ax.tick_params(labelsize=font-2)
        cb.set_label("CFN bonus", fontsize=font, labelpad=8)

    try:
        import wandb
        wandb.log({"heatmap/cfn_difficulty_panels": wandb.Image(fig)}, step=step)
    except Exception:
        pass
    plt.close(fig)

    # optional scatter: compute distance only here
    if add_scatter:
        M = maps[1]  # Closed, has key
        set_world_state(door_open=False, has_key=True)
        _, D = reachability(door_open=False, has_key=True)
        mask = np.isfinite(D) & ~np.isnan(M)
        if np.any(mask):
            xs, ys = D[mask].ravel(), M[mask].ravel()
            fig2 = plt.figure(figsize=(5,4), dpi=120)
            plt.scatter(xs, ys, s=14, alpha=0.85)
            plt.xlabel("Geodesic distance from start (with key)", fontsize=font)
            plt.ylabel("CFN bonus", fontsize=font)
            plt.title("Bonus vs distance", fontsize=font+1)
            plt.tight_layout()
            try:
                import wandb
                wandb.log({"scatter/bonus_vs_distance": wandb.Image(fig2)}, step=step)
            except Exception:
                pass
            plt.close(fig2)
