import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import torch
import gymnasium as gym
from PIL import Image
import imageio
import os
from pathlib import Path
import matplotlib.pyplot as plt
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper


def create_evaluation_gif(agent, env_name="MiniGrid-DoorKey-5x5-v0", 
                         max_steps=300, gif_path="agent_performance.gif", 
                         num_episodes=3, fps=2):
    """
    Create a GIF showing the agent's performance in the environment.
    
    Args:
        agent: Trained DQN agent
        env_name: Name of the MiniGrid environment
        max_steps: Maximum steps per episode
        gif_path: Path to save the GIF
        num_episodes: Number of episodes to record
        fps: Frames per second for the GIF
    """
    
    # Create evaluation environment
    eval_env = gym.make(env_name, render_mode="rgb_array")
    eval_env = FullyObsWrapper(eval_env)
    eval_env = ImgObsWrapper(eval_env)
    
    all_frames = []
    episode_info = []
    
    for episode in range(num_episodes):
        obs_raw, _ = eval_env.reset()
        obs = agent.process_obs(obs_raw)
        
        episode_frames = []
        episode_reward = 0
        episode_steps = 0
        done = False
        
        # Add episode header frame
        header_frame = create_text_frame(f"Episode {episode + 1}", size=(400, 50))
        episode_frames.append(header_frame)
        
        while not done and episode_steps < max_steps:
            # Render the environment
            frame = eval_env.render()
            if frame is not None:
                # Resize frame for better visibility
                frame = resize_frame(frame, target_size=(400, 400))
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
        
        # Add episode summary frame
        status = "SUCCESS!" if terminated else "TIMEOUT" if episode_steps >= max_steps else "FAILED"
        summary_frame = create_text_frame(
            f"Episode {episode + 1} Complete\n"
            f"Status: {status}\n"
            f"Steps: {episode_steps}\n"
            f"Reward: {episode_reward:.2f}",
            size=(400, 200)
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
    
    # Save GIF
    if all_frames:
        imageio.mimsave(gif_path, all_frames, fps=fps)
        print(f"GIF saved to: {gif_path}")
    
    eval_env.close()
    return episode_info


def evaluate_agent_performance(agent, env_name="MiniGrid-DoorKey-5x5-v0", 
                              num_episodes=10, max_steps=300):
    """
    Evaluate agent performance without creating GIF.
    
    Args:
        agent: Trained DQN agent
        env_name: Name of the MiniGrid environment
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
    
    Returns:
        dict: Evaluation metrics
    """
    
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
    print(f"Episodes: {num_episodes}")
    print(f"Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"Mean Length: {metrics['mean_length']:.1f} ± {metrics['std_length']:.1f}")
    print(f"Success Rate: {metrics['success_rate']:.1%} ({success_count}/{num_episodes})")
    print("="*50)
    
    return metrics


def create_text_frame(text, size=(400, 100), bg_color=(255, 255, 255), text_color=(0, 0, 0)):
    """Create a frame with text for the GIF."""
    from PIL import Image, ImageDraw, ImageFont
    
    # Create image
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
    """Resize frame to target size."""
    if frame is None:
        return None
    
    img = Image.fromarray(frame)
    img = img.resize(target_size, Image.Resampling.NEAREST)  # Use nearest neighbor for pixel art
    return np.array(img)


def save_training_progress_gif(agent, env_name, save_dir="./gifs", 
                              checkpoint_steps=[50000, 100000, 200000, 500000]):
    """
    Save GIFs at different training checkpoints to see learning progress.
    Call this function at different points during training.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Determine current training step (you might need to pass this as parameter)
    current_step = getattr(agent, 'current_timestep', 0)
    
    if current_step in checkpoint_steps:
        gif_path = os.path.join(save_dir, f"agent_step_{current_step}.gif")
        create_evaluation_gif(agent, env_name, gif_path=gif_path, num_episodes=2)


# Example usage functions:
def quick_test(agent, env_name="MiniGrid-DoorKey-5x5-v0"):
    """Quick test to see if agent is working."""
    print("Running quick performance test...")
    metrics = evaluate_agent_performance(agent, env_name, num_episodes=5)
    
    if metrics['success_rate'] > 0:
        print("🎉 Agent is successfully solving some episodes!")
        print("Creating demonstration GIF...")
        create_evaluation_gif(agent, env_name, gif_path="quick_demo.gif", num_episodes=2)
    else:
        print("❌ Agent is not yet successful. Keep training!")
    
    return metrics


def full_evaluation(agent, env_name="MiniGrid-DoorKey-5x5-v0"):
    """Full evaluation with GIF creation."""
    print("Running full evaluation...")
    
    # Performance metrics
    metrics = evaluate_agent_performance(agent, env_name, num_episodes=20)
    
    # Create demonstration GIF
    print("Creating demonstration GIF...")
    episode_info = create_evaluation_gif(
        agent, env_name, 
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