from pathlib import Path
import torch
from hydra.core.hydra_config import HydraConfig


def validate(actor, current_timestep, filename=None):
    """
    Save actor checkpoint.

    Args:
        actor: PyTorch actor network
        current_timestep: int, current timestep in training
        filename: Optional[str], custom filename instead of default
    """

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Construct filename
    if filename is None:
        filename = f"sac_actor_step{current_timestep}.pt"
    ckpt_path = ckpt_dir / filename

    # Save actor
    torch.save(actor.state_dict(), ckpt_path)


def validate_dqn(agent, current_timestep, filename=None, save_dir="checkpoints"):
    """
    Save DQN agent checkpoint.

    Args:
        agent: DQN agent with q_net, target_q_net, and optionally cfn
        current_timestep: int, current timestep in training
        filename: Optional[str], custom filename
        save_dir: str, directory to save checkpoints
    """
    ckpt_dir = Path(save_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"dqn_agent_step{current_timestep}.pt"
    ckpt_path = ckpt_dir / filename

    checkpoint = {
        "q_net": agent.q_net.state_dict(),
        "target_q_net": agent.target_q_net.state_dict(),
        "step": current_timestep
    }

    if hasattr(agent, "cfn"):
        checkpoint["cfn"] = agent.cfn.state_dict()

    torch.save(checkpoint, ckpt_path)
    print(f"[VALIDATE] DQN checkpoint saved to {ckpt_path}")
