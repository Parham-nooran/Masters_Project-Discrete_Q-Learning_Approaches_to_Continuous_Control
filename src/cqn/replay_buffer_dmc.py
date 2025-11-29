import datetime
import io
import random
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset


class EpisodeSerializer:
    """Handles serialization and deserialization of episodes."""

    @staticmethod
    def save(episode, file_path):
        """Save episode to compressed numpy file."""
        with io.BytesIO() as bs:
            np.savez_compressed(bs, **episode)
            bs.seek(0)
            with file_path.open("wb") as f:
                f.write(bs.read())

    @staticmethod
    def load(file_path):
        """Load episode from compressed numpy file."""
        with file_path.open("rb") as f:
            episode = np.load(f)
            return {k: episode[k] for k in episode.keys()}

    @staticmethod
    def get_episode_length(episode):
        """Get the length of an episode."""
        return next(iter(episode.values())).shape[0] - 1


class EpisodeBuilder:
    """Builds episodes from individual time steps."""

    def __init__(self, data_specs):
        self.data_specs = data_specs
        self.current_episode = defaultdict(list)

    def add_timestep(self, time_step):
        """Add a single time step to the current episode."""
        for spec in self.data_specs:
            value = self._extract_value(time_step, spec)
            value = self._validate_and_convert(value, spec)
            self.current_episode[spec.name].append(value)

    def _extract_value(self, time_step, spec):
        """Extract value from time step for given spec."""
        return time_step[spec.name]

    def _validate_and_convert(self, value, spec):
        """Validate and convert value to match spec."""
        if np.isscalar(value):
            value = np.full(spec.shape, value, spec.dtype)

        if value.dtype != spec.dtype:
            value = value.astype(spec.dtype)

        if value.shape != spec.shape:
            if len(value.shape) == 0:
                value = np.full(spec.shape, value, spec.dtype)

        return value

    def finalize_episode(self):
        """Finalize and return the current episode."""
        episode = {}
        for spec in self.data_specs:
            value = self.current_episode[spec.name]
            episode[spec.name] = np.array(value, spec.dtype)

        self.current_episode = defaultdict(list)
        return episode

    def is_episode_complete(self, time_step):
        """Check if the episode is complete."""
        return time_step.last()


class EpisodeMetadata:
    """Tracks metadata about stored episodes."""

    def __init__(self):
        self.num_episodes = 0
        self.num_transitions = 0

    def preload_from_directory(self, replay_dir):
        """Preload metadata from existing episode files."""
        for file_path in replay_dir.glob("*.npz"):
            self._add_from_filename(file_path)

    def _add_from_filename(self, file_path):
        """Extract and add metadata from filename."""
        _, _, eps_len = file_path.stem.split("_")
        self.num_episodes += 1
        self.num_transitions += int(eps_len)

    def add_episode(self, episode_length):
        """Add a new episode to metadata."""
        self.num_episodes += 1
        self.num_transitions += episode_length

    def __len__(self):
        return self.num_transitions


class ReplayBufferStorage:
    """Storage for replay buffer episodes on disk."""

    def __init__(self, data_specs, replay_dir):
        self.data_specs = data_specs
        self.replay_dir = Path(replay_dir)
        self.replay_dir.mkdir(exist_ok=True)

        self.episode_builder = EpisodeBuilder(data_specs)
        self.metadata = EpisodeMetadata()
        self.metadata.preload_from_directory(self.replay_dir)

    def add(self, time_step):
        """Add a time step to the replay buffer."""
        self.episode_builder.add_timestep(time_step)

        if self.episode_builder.is_episode_complete(time_step):
            episode = self.episode_builder.finalize_episode()
            self._store_episode(episode)

    def _store_episode(self, episode):
        """Store complete episode to disk."""
        episode_length = EpisodeSerializer.get_episode_length(episode)
        file_path = self._generate_episode_filename(episode_length)

        EpisodeSerializer.save(episode, file_path)
        self.metadata.add_episode(episode_length)

    def _generate_episode_filename(self, episode_length):
        """Generate unique filename for episode."""
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        episode_index = self.metadata.num_episodes
        filename = f"{timestamp}_{episode_index}_{episode_length}.npz"
        return self.replay_dir / filename

    def __len__(self):
        return len(self.metadata)


class NStepReturnCalculator:
    """Calculates n-step returns for reinforcement learning."""

    def __init__(self, nstep, discount):
        self.nstep = nstep
        self.discount = discount

    def calculate(self, episode, start_idx):
        """Calculate n-step return starting from given index."""
        reward = np.zeros_like(episode["reward"][start_idx])
        discount = np.ones_like(episode["discount"][start_idx])

        for i in range(self.nstep):
            step_reward = episode["reward"][start_idx + i]
            reward += discount * step_reward
            discount *= episode["discount"][start_idx + i] * self.discount

        return reward, discount


class TransitionSampler:
    """Samples transitions from episodes."""

    def __init__(self, nstep, discount, low_dim_size):
        self.return_calculator = NStepReturnCalculator(nstep, discount)
        self.low_dim_size = low_dim_size
        self.nstep = nstep

    def sample_transition(self, episode):
        """Sample a single transition from an episode."""
        episode_length = EpisodeSerializer.get_episode_length(episode)
        idx = self._sample_valid_index(episode_length)

        return self._create_transition(episode, idx)

    def _sample_valid_index(self, episode_length):
        """Sample a valid starting index for n-step return."""
        return np.random.randint(0, episode_length - self.nstep + 1) + 1

    def _create_transition(self, episode, idx):
        """Create transition tuple from episode and index."""
        rgb_obs = episode["observation"][idx - 1]
        next_rgb_obs = episode["observation"][idx + self.nstep - 1]

        low_dim_obs = self._create_dummy_low_dim_obs()
        next_low_dim_obs = self._create_dummy_low_dim_obs()

        action = episode["action"][idx]
        reward, discount = self.return_calculator.calculate(episode, idx)

        demos = np.zeros(1, dtype=np.float32)

        return (
            rgb_obs,
            low_dim_obs,
            action,
            reward,
            discount,
            next_rgb_obs,
            next_low_dim_obs,
            demos
        )

    def _create_dummy_low_dim_obs(self):
        """Create dummy low-dimensional observation."""
        return np.zeros(self.low_dim_size, dtype=np.float32)


class EpisodeCache:
    """Manages caching of episodes in memory."""

    def __init__(self, max_size, save_snapshot):
        self.max_size = max_size
        self.save_snapshot = save_snapshot
        self.episodes = {}
        self.episode_files = []
        self.current_size = 0

    def add_episode(self, file_path):
        """Add episode from file to cache."""
        try:
            episode = EpisodeSerializer.load(file_path)
        except Exception:
            return False

        episode_length = EpisodeSerializer.get_episode_length(episode)

        self._ensure_capacity(episode_length)
        self._store_episode(file_path, episode, episode_length)

        if not self.save_snapshot:
            file_path.unlink(missing_ok=True)

        return True

    def _ensure_capacity(self, required_length):
        """Ensure cache has capacity for new episode."""
        while required_length + self.current_size > self.max_size:
            self._remove_oldest_episode()

    def _remove_oldest_episode(self):
        """Remove the oldest episode from cache."""
        if not self.episode_files:
            return

        old_file = self.episode_files.pop(0)
        old_episode = self.episodes.pop(old_file)
        self.current_size -= EpisodeSerializer.get_episode_length(old_episode)
        old_file.unlink(missing_ok=True)

    def _store_episode(self, file_path, episode, episode_length):
        """Store episode in cache."""
        self.episode_files.append(file_path)
        self.episode_files.sort()
        self.episodes[file_path] = episode
        self.current_size += episode_length

    def get_random_episode(self):
        """Get a random episode from cache."""
        file_path = random.choice(self.episode_files)
        return self.episodes[file_path]

    def has_episodes(self):
        """Check if cache has any episodes."""
        return len(self.episode_files) > 0


def _get_episode_length_from_filename(file_path):
    """Extract episode length from filename."""
    parts = file_path.stem.split("_")
    return int(parts[2])


def _get_episode_index_from_filename(file_path):
    """Extract episode index from filename."""
    parts = file_path.stem.split("_")
    return int(parts[1])


def _get_worker_id():
    """Get current worker ID."""
    try:
        return torch.utils.data.get_worker_info().id
    except (AttributeError, RuntimeError):
        return 0


class ReplayBuffer(IterableDataset):
    """Replay buffer that loads episodes from disk and samples transitions."""

    def __init__(
            self,
            replay_dir,
            max_size,
            num_workers,
            nstep,
            discount,
            fetch_every,
            save_snapshot,
            low_dim_size=1,
    ):
        self.replay_dir = Path(replay_dir)
        self.num_workers = max(1, num_workers)
        self.fetch_every = fetch_every
        self.samples_since_last_fetch = fetch_every

        self.episode_cache = EpisodeCache(max_size, save_snapshot)
        self.transition_sampler = TransitionSampler(nstep, discount, low_dim_size)

    def _fetch_new_episodes(self):
        """Fetch new episodes from disk if needed."""
        if self.samples_since_last_fetch < self.fetch_every:
            return

        self.samples_since_last_fetch = 0
        worker_id = _get_worker_id()

        episode_files = self._get_sorted_episode_files()
        self._load_worker_episodes(episode_files, worker_id)

    def _get_sorted_episode_files(self):
        """Get sorted list of episode files."""
        return sorted(self.replay_dir.glob("*.npz"), reverse=True)

    def _load_worker_episodes(self, episode_files, worker_id):
        """Load episodes assigned to this worker."""
        fetched_size = 0

        for file_path in episode_files:
            if not self._should_load_file(file_path, worker_id):
                continue

            if self._is_already_cached(file_path):
                break

            episode_length = _get_episode_length_from_filename(file_path)

            if not self._has_capacity(fetched_size, episode_length):
                break

            if not self.episode_cache.add_episode(file_path):
                break

            fetched_size += episode_length

    def _should_load_file(self, file_path, worker_id):
        """Check if file should be loaded by this worker."""
        episode_index = _get_episode_index_from_filename(file_path)
        return episode_index % self.num_workers == worker_id

    def _is_already_cached(self, file_path):
        """Check if episode is already cached."""
        return file_path in self.episode_cache.episodes

    def _has_capacity(self, fetched_size, episode_length):
        """Check if there's capacity for more episodes."""
        return fetched_size + episode_length <= self.episode_cache.max_size

    def _sample(self):
        """Sample a single transition from the replay buffer."""
        try:
            self._fetch_new_episodes()
        except Exception:
            traceback.print_exc()

        self.samples_since_last_fetch += 1

        episode = self.episode_cache.get_random_episode()
        return self.transition_sampler.sample_transition(episode)

    def __iter__(self):
        """Iterate over replay buffer samples."""
        while True:
            yield self._sample()


def initialize_worker(worker_id):
    """Initialize worker with unique random seed."""
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)


def make_replay_loader(
        replay_dir,
        max_size,
        batch_size,
        num_workers,
        save_snapshot,
        nstep,
        discount,
        low_dim_size=1
):
    """Create a replay buffer data loader."""
    max_size_per_worker = max_size // max(1, num_workers)

    replay_buffer = ReplayBuffer(
        replay_dir,
        max_size_per_worker,
        num_workers,
        nstep,
        discount,
        fetch_every=1000,
        save_snapshot=save_snapshot,
        low_dim_size=low_dim_size,
    )

    loader = torch.utils.data.DataLoader(
        replay_buffer,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=initialize_worker,
    )

    return loader