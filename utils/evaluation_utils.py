import numpy as np
import torch
import gymnasium as gym
from PIL import Image
import imageio
import os
from pathlib import Path
import matplotlib.pyplot as plt
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper


def create_evaluation_gif(agent, max_steps=300, gif_path="agent_performance.gif", 
                         num_episodes=3, fps=2):
    """
    Create a GIF showing the agent's performance in the environment.
    Uses the agent's stored environment name.
    
    Args:
        agent: Trained DQN agent (must have env_name attribute)
        max_steps: Maximum steps per episode
        gif_path: Path to save the GIF
        num_episodes: Number of episodes to record
        fps: Frames per second for the GIF
    """
    
    # Use the agent's environment name
    env_name = agent.env_name
    
    # Create evaluation environment
    eval_env = gym.make(env_name, render_mode="rgb_array")
    eval_env = FullyObsWrapper(eval_env)
    eval_env = ImgObsWrapper(eval_env)
    
    all_frames = []
    episode_info = []
    
    # Standard frame size for consistency
    FRAME_SIZE = (400, 400)
    
    for episode in range(num_episodes):
        obs_raw, _ = eval_env.reset()
        obs = agent.process_obs(obs_raw)
        
        episode_frames = []
        episode_reward = 0
        episode_steps = 0
        done = False
        
        # Add episode header frame with consistent size
        header_frame = create_text_frame(f"Episode {episode + 1}", size=FRAME_SIZE)
        episode_frames.append(header_frame)
        
        while not done and episode_steps < max_steps:
            # Render the environment
            frame = eval_env.render()
            if frame is not None:
                # Resize frame to consistent size
                frame = resize_frame(frame, target_size=FRAME_SIZE)
                episode_frames.append(frame)
            
            # Agent takes action (no exploration during evaluation)
            action = agent.act(obs, epsilon=0.0)  # Greedy action
            
            # Environment step
            next_obs_raw, reward, terminated, truncated, info = eval_env.step(action)
            next_obs = agent.process_obs(next_obs_raw)
            
            done = terminated or truncated
            obs = next_obs
            episode_reward += reward
            episode_steps += 1
        
        # Add episode summary frame with consistent size
        status = "SUCCESS!" if terminated else "TIMEOUT" if episode_steps >= max_steps else "FAILED"
        summary_frame = create_text_frame(
            f"Episode {episode + 1} Complete\n"
            f"Status: {status}\n"
            f"Steps: {episode_steps}\n"
            f"Reward: {episode_reward:.2f}",
            size=FRAME_SIZE  # Use same size as other frames
        )
        episode_frames.append(summary_frame)
        
        # Add some pause frames
        for _ in range(fps):  # 1 second pause
            episode_frames.append(summary_frame)
        
        all_frames.extend(episode_frames)
        episode_info.append({
            'episode': episode + 1,
            'steps': episode_steps,
            'reward': episode_reward,
            'success': terminated
        })
        
        print(f"Episode {episode + 1}: {status}, Steps: {episode_steps}, Reward: {episode_reward:.2f}")
    
    # Ensure all frames have the same shape before saving
    if all_frames:
        # Convert all frames to numpy arrays with consistent shape
        consistent_frames = []
        for frame in all_frames:
            if isinstance(frame, np.ndarray):
                # Ensure frame has correct shape (H, W, 3)
                if len(frame.shape) == 2:  # Grayscale
                    frame = np.stack([frame] * 3, axis=-1)
                elif frame.shape[-1] == 4:  # RGBA
                    frame = frame[:, :, :3]  # Remove alpha channel
                
                # Ensure frame is the right size
                if frame.shape[:2] != FRAME_SIZE:
                    frame = resize_frame(frame, target_size=FRAME_SIZE)
                
                consistent_frames.append(frame.astype(np.uint8))
        
        if consistent_frames:
            imageio.mimsave(gif_path, consistent_frames, fps=fps)
            print(f"GIF saved to: {gif_path}")
        else:
            print("No valid frames to save")
    
    eval_env.close()
    return episode_info


def evaluate_agent_performance(agent, num_episodes=10, max_steps=300):
    """
    Evaluate agent performance without creating GIF.
    Uses the agent's stored environment name.
    
    Args:
        agent: Trained DQN agent (must have env_name attribute)
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
    
    Returns:
        dict: Evaluation metrics
    """
    
    # Use the agent's environment name
    env_name = agent.env_name
    
    eval_env = gym.make(env_name, render_mode="rgb_array")
    eval_env = FullyObsWrapper(eval_env)
    eval_env = ImgObsWrapper(eval_env)
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    for episode in range(num_episodes):
        obs_raw, _ = eval_env.reset()
        obs = agent.process_obs(obs_raw)
        
        episode_reward = 0
        episode_steps = 0
        done = False
        
        while not done and episode_steps < max_steps:
            # Greedy action (no exploration)
            action = agent.act(obs, epsilon=0.0)
            
            next_obs_raw, reward, terminated, truncated, _ = eval_env.step(action)
            next_obs = agent.process_obs(next_obs_raw)
            
            done = terminated or truncated
            obs = next_obs
            episode_reward += reward
            episode_steps += 1
            
            if terminated:  # Successfully completed the task
                success_count += 1
    
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_steps)
    
    eval_env.close()
    
    metrics = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'success_rate': success_count / num_episodes,
        'episodes': episode_rewards,
        'lengths': episode_lengths
    }
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Environment: {env_name}")
    print(f"Episodes: {num_episodes}")
    print(f"Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"Mean Length: {metrics['mean_length']:.1f} ± {metrics['std_length']:.1f}")
    print(f"Success Rate: {metrics['success_rate']:.1%} ({success_count}/{num_episodes})")
    print("="*50)
    
    return metrics


def create_text_frame(text, size=(400, 400), bg_color=(255, 255, 255), text_color=(0, 0, 0)):
    """Create a frame with text for the GIF with consistent size."""
    from PIL import Image, ImageDraw, ImageFont
    
    # Create image with exact size
    img = Image.new('RGB', size, bg_color)
    draw = ImageDraw.Draw(img)
    
    # Try to use a better font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Calculate text position (center)
    lines = text.split('\n')
    total_height = len(lines) * 20
    y_start = (size[1] - total_height) // 2
    
    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        x = (size[0] - text_width) // 2
        y = y_start + i * 20
        draw.text((x, y), line, fill=text_color, font=font)
    
    return np.array(img)


def resize_frame(frame, target_size=(400, 400)):
    """Resize frame to target size maintaining aspect ratio and ensuring correct format."""
    if frame is None:
        return None
    
    # Convert to PIL Image
    if isinstance(frame, np.ndarray):
        # Ensure frame is uint8
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
        
        # Handle different channel configurations
        if len(frame.shape) == 2:  # Grayscale
            img = Image.fromarray(frame, mode='L').convert('RGB')
        elif len(frame.shape) == 3:
            if frame.shape[2] == 3:  # RGB
                img = Image.fromarray(frame, mode='RGB')
            elif frame.shape[2] == 4:  # RGBA
                img = Image.fromarray(frame, mode='RGBA').convert('RGB')
            else:
                # Handle unexpected channel count
                img = Image.fromarray(frame[:, :, 0], mode='L').convert('RGB')
        else:
            raise ValueError(f"Unexpected frame shape: {frame.shape}")
    else:
        img = frame
    
    # Resize with nearest neighbor for pixel art (better for MiniGrid)
    img = img.resize(target_size, Image.Resampling.NEAREST)
    
    # Convert back to numpy array
    result = np.array(img)
    
    # Ensure result has shape (H, W, 3)
    if len(result.shape) == 2:
        result = np.stack([result] * 3, axis=-1)
    elif result.shape[2] != 3:
        result = result[:, :, :3]
    
    return result.astype(np.uint8)


def save_training_progress_gif(agent, save_dir="./gifs", 
                              checkpoint_steps=[50000, 100000, 200000, 500000]):
    """
    Save GIFs at different training checkpoints to see learning progress.
    Uses the agent's stored environment name.
    Call this function at different points during training.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Determine current training step (you might need to pass this as parameter)
    current_step = getattr(agent, 'current_timestep', 0)
    
    if current_step in checkpoint_steps:
        gif_path = os.path.join(save_dir, f"agent_step_{current_step}.gif")
        create_evaluation_gif(agent, gif_path=gif_path, num_episodes=2)


def diagnose_cfn_issues(agent, num_samples=100):
    """
    Diagnose potential issues with CFN exploration.
    Uses the agent's stored environment name.
    """
    print("\n" + "="*50)
    print("CFN DIAGNOSTIC")
    print("="*50)
    
    # Use the agent's environment name
    env_name = agent.env_name
    
    # Sample some observations from the environment
    env = gym.make(env_name, render_mode="rgb_array")
    env = FullyObsWrapper(env)
    env = ImgObsWrapper(env)
    
    intrinsic_rewards = []
    output_norms = []
    
    for _ in range(num_samples):
        obs_raw, _ = env.reset()
        obs = agent.process_obs(obs_raw)
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(agent.device)
        
        with torch.no_grad():
            # Get CFN output
            cfn_output = agent.cfn(obs_tensor, update_prior_stats=False)
            output_norm = cfn_output.norm(p=2, dim=1).item()
            
            # Compute intrinsic reward
            from CFN.priority_util import compute_intrinsic_reward
            intrinsic_reward = compute_intrinsic_reward(
                agent.coin_flip_dim,
                torch.tensor(output_norm ** 2)
            ).item()
            
            intrinsic_rewards.append(intrinsic_reward)
            output_norms.append(output_norm)
    
    env.close()
    
    print(f"Environment: {env_name}")
    print(f"Intrinsic Rewards - Mean: {np.mean(intrinsic_rewards):.4f}, Std: {np.std(intrinsic_rewards):.4f}")
    print(f"Output Norms - Mean: {np.mean(output_norms):.4f}, Std: {np.std(output_norms):.4f}")
    print(f"Min/Max Intrinsic: {np.min(intrinsic_rewards):.4f} / {np.max(intrinsic_rewards):.4f}")
    
    # Check if intrinsic rewards are too low/high
    mean_intrinsic = np.mean(intrinsic_rewards)
    if mean_intrinsic < 0.1:
        print("⚠️  WARNING: Intrinsic rewards very low - agent might not explore enough")
    elif mean_intrinsic > 10:
        print("⚠️  WARNING: Intrinsic rewards very high - might overwhelm external rewards")
    else:
        print("✅ Intrinsic reward range looks reasonable")
    
    print("="*50)


def debug_training_progress(agent):
    """Debug why agent isn't learning. Uses agent's environment name."""
    print("\n" + "="*50)
    print("TRAINING DEBUG")
    print("="*50)
    print(f"Environment: {agent.env_name}")
    
    # Check if DQN is actually updating
    q_net_params = list(agent.q_net.parameters())
    if len(q_net_params) > 0:
        total_grad_norm = 0
        param_count = 0
        for param in q_net_params:
            if param.grad is not None:
                total_grad_norm += param.grad.data.norm(2).item()
                param_count += 1
        
        if param_count > 0:
            avg_grad_norm = total_grad_norm / param_count
            print(f"DQN Gradient Norm: {avg_grad_norm:.6f}")
            if avg_grad_norm < 1e-6:
                print("⚠️  WARNING: Very small gradients - learning might be stuck")
        else:
            print("⚠️  WARNING: No gradients found - DQN not updating")
    
    # Check replay buffer
    buffer_size = len(agent.replay_buffer)
    print(f"Replay Buffer Size: {buffer_size}")
    if buffer_size < agent.batch_size:
        print("⚠️  WARNING: Replay buffer too small for learning")
    
    # Check CFN buffer
    cfn_buffer_size = agent.cfn_buffer.size
    print(f"CFN Buffer Size: {cfn_buffer_size}")
    
    # Sample from replay buffer to check reward distribution
    if buffer_size >= agent.batch_size:
        import random
        sample = random.sample(agent.replay_buffer, min(100, buffer_size))
        rewards = [transition[2] for transition in sample]  # reward is index 2
        print(f"Recent Rewards - Mean: {np.mean(rewards):.3f}, Std: {np.std(rewards):.3f}")
        print(f"Reward Range: {np.min(rewards):.3f} to {np.max(rewards):.3f}")
        
        # Count zero rewards
        zero_count = sum(1 for r in rewards if abs(r) < 0.01)
        print(f"Zero/Near-zero rewards: {zero_count}/100 ({zero_count}%)")
    
    print("="*50)


def quick_test(agent):
    """Quick test to see if agent is working. Uses agent's environment name."""
    print("Running quick performance test...")
    metrics = evaluate_agent_performance(agent, num_episodes=5)
    
    if metrics['success_rate'] > 0:
        print("🎉 Agent is successfully solving some episodes!")
        print("Creating demonstration GIF...")
        create_evaluation_gif(agent, gif_path="quick_demo.gif", num_episodes=2)
    else:
        print("❌ Agent is not yet successful. Keep training!")
    
    return metrics


def full_evaluation(agent):
    """Full evaluation with GIF creation. Uses agent's environment name."""
    print("Running full evaluation...")
    
    # Performance metrics
    metrics = evaluate_agent_performance(agent, num_episodes=20)
    
    # Create demonstration GIF
    print("Creating demonstration GIF...")
    episode_info = create_evaluation_gif(
        agent, 
        gif_path="full_evaluation.gif", 
        num_episodes=5, 
        fps=3
    )
    
    # Create performance plot
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.hist(metrics['episodes'], bins=10, alpha=0.7)
    plt.xlabel('Episode Reward')
    plt.ylabel('Frequency')
    plt.title('Reward Distribution')
    
    plt.subplot(1, 3, 2)
    plt.hist(metrics['lengths'], bins=10, alpha=0.7)
    plt.xlabel('Episode Length')
    plt.ylabel('Frequency')
    plt.title('Episode Length Distribution')
    
    plt.subplot(1, 3, 3)
    success_data = [metrics['success_rate'], 1 - metrics['success_rate']]
    plt.pie(success_data, labels=['Success', 'Failure'], autopct='%1.1f%%')
    plt.title('Success Rate')
    
    plt.tight_layout()
    plt.savefig('evaluation_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return metrics, episode_info