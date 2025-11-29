import os
import torch


class CheckpointManager:
    """Manages checkpoint loading and saving operations."""

    def __init__(self, logger, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        self.logger = logger
        os.makedirs(checkpoint_dir, exist_ok=True)

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

    def save_checkpoint(self, agent, episode, task_name, seed):
        """Save agent checkpoint.

        Args:
            agent: The trainer object (CQNTrainer) with get_checkpoint_state method
            episode: Current episode number
            task_name: Name of the task
            seed: Random seed

        Returns:
            Path to saved checkpoint
        """
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"{task_name}_{seed}_{episode}.pth"
        )

        # Get checkpoint state from the trainer
        if hasattr(agent, 'get_checkpoint_state'):
            checkpoint_state = agent.get_checkpoint_state()
        else:
            # Fallback for simple agent objects
            checkpoint_state = {
                'agent_state_dict': agent.state_dict() if hasattr(agent, 'state_dict') else None,
                'episode': episode,
            }

        torch.save(checkpoint_state, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

        return checkpoint_path

    def load_checkpoint(self, checkpoint_path, agent):
        """Load checkpoint into agent.

        Args:
            checkpoint_path: Path to checkpoint file
            agent: The trainer object to load checkpoint into

        Returns:
            Starting episode number, or 0 if loading failed
        """
        try:
            checkpoint = torch.load(checkpoint_path)

            if hasattr(agent, 'load_checkpoint_state'):
                agent.load_checkpoint_state(checkpoint)
            elif hasattr(agent, 'load_state_dict'):
                agent.load_state_dict(checkpoint.get('agent_state_dict', checkpoint))

            episode = checkpoint.get('global_episode', checkpoint.get('episode', 0))

            self.logger.info(
                f"Loaded checkpoint from {checkpoint_path}, episode {episode}"
            )
            return episode + 1

        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            self.logger.info("Starting fresh training...")
            return 0

    def load_checkpoint_if_available(self, checkpoints, agent) -> int:
        """
        Load checkpoint if specified or find latest.

        Args:
            checkpoints: Path to specific checkpoint or None
            agent: Agent to load checkpoint into.

        Returns:
            Starting episode number.
        """
        if checkpoints:
            return self.load_checkpoint(checkpoints, agent)

        latest = self.find_latest_checkpoint()
        if latest:
            self.logger.info(f"Found latest checkpoint: {latest}")
            return self.load_checkpoint(latest, agent)

        return 0