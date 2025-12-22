import random
import numpy as np
import torch
from cpprb import PrioritizedReplayBuffer

from CFN.priority_util import compute_cfn_priority
from cpprb import PrioritizedReplayBuffer
import numpy as np
import torch

from collections import defaultdict
import numpy as np
import torch
from cpprb import PrioritizedReplayBuffer

from collections import defaultdict
import numpy as np
import torch
from cpprb import PrioritizedReplayBuffer


class CFNReplayBufferWrapper:
    def __init__(self, size, obs_shape, coin_flip_dim, alpha=0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.buffer = PrioritizedReplayBuffer(
            size,
            env_dict={
                "obs":       {"shape": obs_shape,        "dtype": np.uint8},
                "coin_flip": {"shape": (coin_flip_dim,), "dtype": np.float32},
            }
        )

        self.size = size
        self.alpha = alpha

        # nupdates(i): number of times instance i has been sampled
        self.counters = np.zeros(size, dtype=np.float32)

    def get_stored_size(self):
        return self.buffer.get_stored_size()

    def add(self, obs, coin_flip, priority):
        """
        Add a new (state, coin_flip) instance.
        Counters are NOT touched here.
        """
        if obs.dtype != np.uint8:
            obs = obs.astype(np.uint8, copy=False)
        if coin_flip.dtype != np.float32:
            coin_flip = coin_flip.astype(np.float32, copy=False)

        self.buffer.add(obs=obs, coin_flip=coin_flip, priority=priority)

    def sample_with_indices(self, batch_size):
        sample = self.buffer.sample(batch_size)
        indices = sample["indexes"]

        obs_batch = torch.from_numpy(sample["obs"]).float().to(self.device)
        coin_flip_batch = torch.from_numpy(sample["coin_flip"]).float().to(self.device)

        return obs_batch, coin_flip_batch, indices

    def update_priorities(self, indices, obs_batch, cfn, coin_flip_dim):
        """
        Update nupdates and PER priorities according to Eq. (6) in the paper.
        """

        # i am updating the instance counter instead of the state counter (according to the paper)
        np.add.at(self.counters, indices, 1.0)

        # fetch counts for sampled instances
        counts = torch.from_numpy(self.counters[indices]).float().to(obs_batch.device)

        new_priorities = compute_cfn_priority(
            cfn,
            obs_batch,
            counts,
            coin_flip_dim,
            alpha=self.alpha
        )

        self.buffer.update_priorities(
            indices,
            new_priorities.detach().cpu().numpy()
        )

    def sample_and_update_priorities(
        self,
        batch_size,
        cfn,
        coin_flip_dim,
        use_cfn_priority=True
    ):
        sample = self.buffer.sample(batch_size)
        indices = sample["indexes"]

        obs_batch = torch.from_numpy(sample["obs"]).float().to(self.device)
        coin_flip_batch = torch.from_numpy(sample["coin_flip"]).float().to(self.device)

        if use_cfn_priority:
            np.add.at(self.counters, indices, 1.0)
            counts = torch.from_numpy(self.counters[indices]).float().to(self.device)

            new_priorities = compute_cfn_priority(
                cfn,
                obs_batch,
                counts,
                coin_flip_dim,
                alpha=self.alpha
            )

            self.buffer.update_priorities(
                indices,
                new_priorities.detach().cpu().numpy()
            )

        return obs_batch, coin_flip_batch, indices



class CFNReplayBuffer:
    def __init__(self, max_size: int):
        """
        Create a buffer for storing state and coin-flip pairs.

        :param max_size: Maximum number of (state, coin_flip) pairs in the buffer.
        """
        self.data = []
        self.sample_counts = []
        self.max_size = max_size
        self.position = 0

    def __len__(self) -> int:
        """Returns how many state-coin flip pairs are currently in the buffer."""
        return len(self.data)

    def store(self, state: torch.Tensor, coin_flip: torch.Tensor):
        """
        Adds a new (state, coin_flip) pair to the buffer. If the buffer is full,
        it overwrites the oldest pair.

        :param state: The current state.
        :param coin_flip: The coin flip vector associated with the state.
        """
        if len(self.data) < self.max_size:
            self.data.append((state, coin_flip, 0, 1))
        else:
            self.data[self.position] = (state, coin_flip, 0, 1)
        self.position = (self.position + 1) % self.max_size

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
        """
        Sample a batch of (state, coin_flip, count) tuples uniformly with replacement.

        :param batch_size: The batch size.
        :returns: A tuple (state_batch, coin_flip_batch, count_batch, indices).

        """

        raw_priorities = [entry[3] for entry in self.data]
        priority_sum = sum(raw_priorities)

        if priority_sum == 0:
            # fallback to uniform if all priorities are 0
            probs = [1 / len(self.data)] * len(self.data)
        else:
            probs = [p / priority_sum for p in raw_priorities]

        indices = random.choices(range(len(self.data)), weights=probs, k=batch_size)

        states, coin_flips, counts, priorities = [], [], [], []

        for i in indices:
            state, coin, count, priority = self.data[i]
            self.data[i] = (state, coin, count + 1, priority)
            states.append(state)
            coin_flips.append(coin)
            counts.append(count + 1)

        return (
            torch.stack(states),
            torch.stack(coin_flips),
            torch.tensor(counts),
            indices
        )

    def update_priorities(self, indices: list[int], new_priorities: list[float]):
        for i, p in zip(indices, new_priorities):
            state, coin, count, _ = self.data[i]
            self.data[i] = (state, coin, count, p)
