from collections import namedtuple

EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards", "timesteps_on_ep_end"])
