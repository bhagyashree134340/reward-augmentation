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
