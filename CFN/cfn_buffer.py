import torch
import random


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




