# evaluate_any.py
import os
import torch
import gymnasium as gym
import imageio
from pathlib import Path
from minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper, ImgObsWrapper
from agents import customised_doorkey
from networks.ddqn import DDQN

# ===== HARD-CODED CHECKPOINT PATH (env var override optional) =====
CHECKPOINT_PATH = os.getenv("DQN_CKPT", "checkpoints_rnd/dqn_agent_step500000.pt")

# ===== ENV SETUP (must match training wrappers) =====
def make_env():
    env = gym.make(
        "Fixed-DoorKey-6x6-v0",
        disable_env_checker=True,
        render_mode="rgb_array",
        key_pos=(1, 4),
        door_pos=(3, 3),
        goal_pos=(4, 3),
        agent_start_pos=(1, 1),
    )
    env = customised_doorkey.NoDropWrapper(env)
    env = FullyObsWrapper(env)
    env = RGBImgObsWrapper(env, tile_size=4)
    env = ImgObsWrapper(env)
    return env

def _extract_qnet_state(ckpt):
    """
    Accepts:
      - {"q_net": state_dict, "target_q_net": ..., "step": ...}
      - {"state_dict": state_dict}
      - bare state_dict (mapping of param names -> tensors)
    Returns:
      - state_dict for the Q-network
    """
    if isinstance(ckpt, dict):
        if "q_net" in ckpt and isinstance(ckpt["q_net"], dict):
            return ckpt["q_net"]
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
        # Heuristic: looks like a bare state_dict
        if all(isinstance(k, str) for k in ckpt.keys()):
            return ckpt
    # If someone saved a whole object with .q_net
    if hasattr(ckpt, "q_net") and hasattr(ckpt.q_net, "state_dict"):
        return ckpt.q_net.state_dict()
    raise RuntimeError(f"Unrecognized checkpoint format: keys={list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}")

def load_model(env, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_shape = env.observation_space.shape     # H, W, C
    obs_shape_torch = (obs_shape[2], obs_shape[0], obs_shape[1])  # C, H, W
    act_dim = env.action_space.n

    model = DDQN(obs_shape_torch, act_dim, hidden_size=128, is_cnn=True).to(device)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _extract_qnet_state(ckpt)

    # Handle possible "module." prefixes if saved under DataParallel
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, device

def run_episode_gif(env, model, device, filename):
    frames = []
    obs, _ = env.reset()
    obs = obs.transpose(2, 0, 1)  # HWC -> CHW
    done = False
    while not done:
        # frame = env.render()
        frame = env.unwrapped.render(highlight=False)  # each time you render
        frames.append(frame)

        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0) / 255.0
        with torch.no_grad():
            action = model(obs_tensor).argmax(dim=1).item()

        obs_next, _, terminated, truncated, _ = env.step(action)
        obs = obs_next.transpose(2, 0, 1)
        done = terminated or truncated

    frame = env.unwrapped.render(highlight=False)
    frames.append(frame)

    imageio.mimsave(filename, frames, duration=0.08)  

def main():
    print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
    # Base env for shape/model
    base_env = make_env()
    model, device = load_model(base_env, CHECKPOINT_PATH)

    save_dir = Path("evaluate_dqn_outputs")
    for i in range(5):
        env = make_env()
        gif_path = save_dir / f"minigrid_rnd_{i}.gif"
        run_episode_gif(env, model, device, gif_path)
        print(f"Saved {gif_path}")
        env.close()

    base_env.close()

if __name__ == "__main__":
    main()