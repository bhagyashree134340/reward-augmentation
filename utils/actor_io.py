import torch


def save_actor(actor, save_path):
    torch.save(actor.state_dict(), save_path)


def load_actor(actor, load_path):
    actor.load_state_dict(torch.load(load_path))
    actor.eval()
