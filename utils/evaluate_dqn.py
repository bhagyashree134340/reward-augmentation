import os
import torch
import gymnasium as gym
import imageio
import numpy as np
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper
from agents.dqn_vanilla import DQNAgent  # Ensure this matches your actual path

def create_eval_gif_dqn():
    # === Config ===
    checkpoint_path = "checkpoints/dqn_agent_step970000.pt"
    gif_path = "evaluate_dqn_outputs/rollout_step_970000.gif"
    env_name = "MiniGrid-DoorKey-6x6-v0"
    max_steps = 1250

    # === Setup Env ===
    eval_env = gym.make(env_name, render_mode="rgb_array", max_episode_steps=max_steps)
    eval_env = FullyObsWrapper(eval_env)
    eval_env = ImgObsWrapper(eval_env)

    # Dummy env for agent init
    train_env = gym.make(env_name, render_mode="rgb_array", max_episode_steps=max_steps)
    train_env = FullyObsWrapper(train_env)
    train_env = ImgObsWrapper(train_env)

    # === Agent Config ===
    cfg = type("DQNConfig", (), {
        "hidden_size": 256,
        "lr": 1e-5,
        "gamma": 0.99,
        "batch_size": 128,
        "replay_buffer_size": 1_000_000,
        "target_update_freq": 2000,
    })

    # === Init agent ===
    agent = DQNAgent(train_env, eval_env, env_name, cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.device = device
    agent.q_net.to(device)

    # === Load q_net from checkpoint dict ===
    checkpoint = torch.load(checkpoint_path, map_location=device)
    agent.q_net.load_state_dict(checkpoint["q_net"])
    agent.q_net.eval()

    # === Rollout ===
    obs_raw, _ = eval_env.reset()
    obs = agent.process_obs(obs_raw)
    frames = []
    done = False

    for step in range(max_steps):
        frame = eval_env.render()
        frames.append(frame)

        # Debug print: current position or step info
        if hasattr(eval_env, "agent_pos"):
            print(f"Step {step} | Agent position: {eval_env.agent_pos}")
        else:
            print(f"Step {step}")

        # Process obs and act
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q_values = agent.q_net(obs_tensor)
            action = torch.argmax(q_values, dim=1).item()

        print(f"Action taken: {action}")

        next_obs_raw, reward, terminated, truncated, info = eval_env.step(action)
        obs = agent.process_obs(next_obs_raw)

        if terminated or truncated:
            final_frame = eval_env.render()
            frames.append(final_frame)
            print(f"Episode ended at step {step} (terminated={terminated}, truncated={truncated})")
            break

    # === Save GIF ===
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    imageio.mimsave(gif_path, frames, fps=10)
    print(f"[GIF] Saved rollout to {gif_path}")


if __name__ == "__main__":
    create_eval_gif_dqn()