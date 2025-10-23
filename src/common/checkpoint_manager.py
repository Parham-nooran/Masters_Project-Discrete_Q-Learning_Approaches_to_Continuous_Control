import os


def load_checkpoint(agent, checkpoint_path, logger):
    """Load checkpoint and return starting episode."""
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        if checkpoint_path:
            logger.warn(f"Checkpoint {checkpoint_path} not found. Starting fresh...")
        return 0

    try:
        loaded_episode = agent.load_checkpoint(checkpoint_path)
        start_episode = loaded_episode + 1
        logger.info(
            f"Resumed from episode {loaded_episode}, starting at {start_episode}"
        )
        return start_episode
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        logger.info("Starting fresh training...")
        return 0


class CheckpointManager:
    """Manages checkpoint loading and saving operations."""

    def __init__(self, logger, checkpoint_dir="output/checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        self.logger = logger

    def find_latest_checkpoint(self):
        """Find the most recent checkpoint file."""
        if not os.path.exists(self.checkpoint_dir):
            return None

        checkpoint_files = [
            f for f in os.listdir(self.checkpoint_dir) if f.endswith(".pth")
        ]

        if not checkpoint_files:
            return None

        checkpoint_files.sort(
            key=lambda x: os.path.getmtime(os.path.join(self.checkpoint_dir, x)),
            reverse=True,
        )

        return os.path.join(self.checkpoint_dir, checkpoint_files[0])

    def save_checkpoint(self, agent, episode, task_name):
        """Save agent checkpoint."""
        checkpoint_path = f"{self.checkpoint_dir}/decqn_{task_name}_{episode}.pth"
        agent.save_checkpoint(checkpoint_path, episode)
        return checkpoint_path

    def load_checkpoint_if_available(self, checkpoints, agent) -> int:
        """
        Load checkpoint if specified or find latest.

        Args:
            agent: Agent to load checkpoint into.

        Returns:
            Starting episode number.
        """
        if checkpoints:
            return load_checkpoint(agent, checkpoints, self.logger)
        latest = self.find_latest_checkpoint()
        if latest:
            self.logger.info(f"Found latest checkpoint: {latest}")
            return load_checkpoint(agent, latest, self.logger)
        return 0
