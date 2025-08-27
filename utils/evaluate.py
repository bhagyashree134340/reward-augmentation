import os
import imageio
import numpy as np
import torch

def evaluate_dqn(agent, eval_env, step, save_dir="eval", num_episodes=10,
                 log_to_wandb=True, fps=6, epsilon_eval=None):
    """
    Evaluate the agent for `num_episodes`.
    - epsilon_eval: if None, uses agent.eval_epsilon (paper gridworld: 0.001).
                    set to 0.0 for fully greedy.
    """
    os.makedirs(save_dir, exist_ok=True)
    returns, lengths = [], []

    if epsilon_eval is None:
        epsilon_eval = getattr(agent, "eval_epsilon", 0.0)

    # switch model to eval mode
    was_training = agent.q_net.training
    agent.q_net.eval()

    with torch.no_grad():
        for ep in range(num_episodes):
            obs_raw, _ = eval_env.reset()
            obs = agent.process_obs(obs_raw)

            done = False
            total_return = 0.0
            ep_len = 0

            frames = []
            record_gif = (ep == 0)
            if record_gif:
                frames.append(eval_env.render())

            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0) / 255.0
                action = agent.act(obs_t, epsilon=epsilon_eval)

                next_obs_raw, reward, terminated, truncated, _ = eval_env.step(action)
                done = bool(terminated or truncated)
                obs = agent.process_obs(next_obs_raw)

                total_return += float(reward)
                ep_len += 1

                if record_gif:
                    frames.append(eval_env.render())

            returns.append(total_return)
            lengths.append(ep_len)

            if record_gif:
                gif_path = os.path.join(save_dir, f"eval_step{step}_ep{ep}.gif")
                imageio.mimsave(gif_path, frames, fps=fps)
                if log_to_wandb:
                    import wandb
                    wandb.log({f"eval/gif_episode_{ep}": wandb.Video(gif_path, fps=fps, format="gif")}, step=step)

    # restore mode
    if was_training:
        agent.q_net.train()

    returns = np.array(returns, dtype=np.float32)
    lengths = np.array(lengths, dtype=np.int32)
    np.savez(os.path.join(save_dir, f"eval_step_{step}.npz"), returns=returns, lengths=lengths)

    if log_to_wandb:
        import wandb
        wandb.log({
            "eval/mean_return": float(returns.mean()),
            "eval/std_return":  float(returns.std()),
            "eval/mean_length": float(lengths.mean()),
            "eval/epsilon": float(epsilon_eval),
        }, step=step)

    print(f"[EVAL] step {step} | avg return: {returns.mean():.3f} | avg len: {lengths.mean():.1f} | eps={epsilon_eval}")
