import numpy as np

from src.bangbang.algorithms import *
from src.bangbang.bernoulli_policy import BernoulliPolicy
from src.common.encoder import VisionEncoder
from src.common.replay_buffer import PrioritizedReplayBuffer
from src.common.training_utils import (
    get_batch_components,
    encode_observation,
    check_and_sample_batch_from_replay_buffer,
)


class BangBangAgent:

    def __init__(
            self,
            config,
            obs_shape: tuple,
            action_spec: dict
    ):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.training_step = 0
        self.last_obs = None

        self._setup_action_space(action_spec)
        self._setup_networks(config, obs_shape)
        self._setup_optimizers(config)
        self.replay_buffer = self._create_replay_buffer(config)
        self.algorithm = self._create_algorithm(config)

    def _setup_action_space(self, action_spec: dict):
        self.action_spec = action_spec
        self.action_dim = len(action_spec["low"])

        # Store action bounds for reference
        self.action_low = torch.tensor(action_spec["low"], device=self.device)
        self.action_high = torch.tensor(action_spec["high"], device=self.device)

    def _setup_networks(self, config, obs_shape: tuple):
        self.encoder, self.encoder_output_size = self._create_encoder(config, obs_shape)
        self.policy = self._create_policy(config)
        self.value_function = self._create_value_function(config)

    def _setup_optimizers(self, config):
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        self.value_optimizer = optim.Adam(self.value_function.parameters(), lr=config.learning_rate)
        self.encoder_optimizer = self._create_encoder_optimizer(config)

    def _create_encoder(self, config, obs_shape: tuple) -> Tuple[Optional[VisionEncoder], int]:
        if config.use_pixels:
            encoder = VisionEncoder(config, config.num_pixels).to(self.device)
            return encoder, config.layer_size_bottleneck
        return None, np.prod(obs_shape)

    def _create_encoder_optimizer(self, config) -> Optional[optim.Adam]:
        if self.encoder:
            return optim.Adam(self.encoder.parameters(), lr=config.learning_rate)
        return None

    def _create_policy(self, config) -> BernoulliPolicy:
        return BernoulliPolicy(
            self.encoder_output_size,
            self.action_dim,
            config.layer_size_network
        ).to(self.device)

    def _create_value_function(self, config) -> LayerNormMLP:
        layer_sizes = [self.encoder_output_size] + config.layer_size_network + [1]
        return LayerNormMLP(layer_sizes, activate_final=False).to(self.device)

    def _create_replay_buffer(self, config) -> PrioritizedReplayBuffer:
        buffer = PrioritizedReplayBuffer(
            capacity=config.max_replay_size,
            alpha=config.priority_exponent,
            beta=config.importance_sampling_exponent,
            n_step=config.adder_n_step,
            discount=config.discount,
        )
        buffer.device = self.device
        return buffer

    def _create_algorithm(self, config) -> Base:
        algorithm_type = getattr(config, "algorithm", "ppo").lower()

        if algorithm_type == "ppo":
            return PPO(
                clip_ratio=getattr(config, "ppo_clip_ratio", 0.2),
                value_coef=getattr(config, "ppo_value_coef", 0.5)
            )
        elif algorithm_type == "sac":
            sac = SAC(
                alpha=getattr(config, "sac_alpha", 0.2),
                tau=getattr(config, "sac_tau", 0.005),
                learning_rate=config.learning_rate
            )
            sac.initialize_q_networks(self.encoder_output_size + self.action_dim, self.device)
            return sac
        elif algorithm_type == "mpo":
            return MPO(
                epsilon=getattr(config, "mpo_epsilon", 0.1),
                epsilon_penalty=getattr(config, "mpo_epsilon_penalty", 0.001)
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_type}")

    def _encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if self.encoder:
            return self.encoder(obs)
        return obs.flatten(1)

    def _binary_to_bangbang(self, binary_action: torch.Tensor) -> torch.Tensor:
        """Convert binary action (0,1) to bang-bang action (-1,1)."""
        return 2.0 * binary_action - 1.0

    def _bangbang_to_binary(self, bangbang_action: torch.Tensor) -> torch.Tensor:
        """Convert bang-bang action (-1,1) to binary action (0,1)."""
        return (bangbang_action + 1.0) / 2.0

    def observe_first(self, obs: torch.Tensor):
        self.last_obs = obs

    def select_action(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        obs = self._prepare_observation(obs)

        with torch.no_grad():
            encoded_obs = self._encode_obs(obs)
            # Policy returns bang-bang actions (-1, 1)
            action, _ = self.policy.get_action(encoded_obs, deterministic)

        return action.squeeze(0)

    def _prepare_observation(self, obs: torch.Tensor) -> torch.Tensor:
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().to(self.device)

        if len(obs.shape) == 1 or (self.encoder and len(obs.shape) == 3):
            obs = obs.unsqueeze(0)

        return obs

    def observe(self, action: torch.Tensor, reward: float, next_obs: torch.Tensor, done: bool):
        if self.last_obs is None:
            return

        # Convert bang-bang action (-1, 1) to binary (0, 1) for storage
        binary_action = self._bangbang_to_binary(action)

        self.replay_buffer.add(self.last_obs, binary_action, reward, next_obs, done)
        self.last_obs = next_obs.detach()

    def update(self) -> Dict[str, float]:
        batch = self._sample_batch()
        if batch is None:
            return {}

        components = get_batch_components(batch, self.device)
        obs, actions, rewards, next_obs, dones, discounts, weights, indices = components

        obs_encoded, next_obs_encoded = encode_observation(self.encoder, obs, next_obs)

        # Ensure actions are properly binary (0 or 1)
        # This handles any potential floating point errors
        actions = torch.round(actions).clamp(0, 1)

        loss, metrics, td_errors = self.algorithm.compute_loss(
            self.policy,
            self.value_function,
            obs_encoded,
            actions,
            rewards,
            next_obs_encoded,
            dones,
            discounts,
            weights,
        )

        self._perform_gradient_update(loss)
        self.algorithm.update_target_networks()
        self._update_replay_priorities(indices, td_errors)

        self.training_step += 1
        return metrics

    def _sample_batch(self) -> Optional[dict]:
        return check_and_sample_batch_from_replay_buffer(
            self.replay_buffer,
            self.config.min_replay_size,
            self.config.batch_size
        )

    def _perform_gradient_update(self, loss: torch.Tensor):
        optimizers = [self.policy_optimizer, self.value_optimizer]
        if self.encoder_optimizer:
            optimizers.append(self.encoder_optimizer)
        optimizers.extend(self.algorithm.get_optimizers())

        for optimizer in optimizers:
            optimizer.zero_grad()

        loss.backward()

        if getattr(self.config, "clip_gradients", False):
            self._clip_gradients()

        for optimizer in optimizers:
            optimizer.step()

    def _clip_gradients(self):
        clip_norm = getattr(self.config, "clip_gradients_norm", 40.0)
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), clip_norm)
        torch.nn.utils.clip_grad_norm_(self.value_function.parameters(), clip_norm)
        if self.encoder:
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), clip_norm)

    def _update_replay_priorities(self, indices, td_errors):
        if td_errors is not None:
            priorities = td_errors.cpu().numpy()
            self.replay_buffer.update_priorities(indices, priorities)

    def save_checkpoint(self, path: str, episode: int):
        checkpoint = self._build_checkpoint(episode)
        torch.save(checkpoint, path)

    def _build_checkpoint(self, episode: int) -> dict:
        checkpoint = {
            "episode": episode,
            "training_step": self.training_step,
            "policy_state_dict": self.policy.state_dict(),
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
            "value_state_dict": self.value_function.state_dict(),
            "value_optimizer_state_dict": self.value_optimizer.state_dict(),
            "action_low": self.action_low.cpu(),
            "action_high": self.action_high.cpu(),
            "config": self.config,
        }

        if self.encoder:
            checkpoint["encoder_state_dict"] = self.encoder.state_dict()
            checkpoint["encoder_optimizer_state_dict"] = self.encoder_optimizer.state_dict()

        if isinstance(self.algorithm, SAC):
            checkpoint.update(self._get_sac_checkpoint_data())

        return checkpoint

    def _get_sac_checkpoint_data(self) -> dict:
        return {
            "q1_state_dict": self.algorithm.q1.state_dict(),
            "q2_state_dict": self.algorithm.q2.state_dict(),
            "target_q1_state_dict": self.algorithm.target_q1.state_dict(),
            "target_q2_state_dict": self.algorithm.target_q2.state_dict(),
            "q1_optimizer_state_dict": self.algorithm.q1_optimizer.state_dict(),
            "q2_optimizer_state_dict": self.algorithm.q2_optimizer.state_dict(),
        }

    def load_checkpoint(self, path: str) -> int:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self._load_base_checkpoint(checkpoint)

        if self.encoder and "encoder_state_dict" in checkpoint:
            self._load_encoder_checkpoint(checkpoint)

        if isinstance(self.algorithm, SAC) and "q1_state_dict" in checkpoint:
            self._load_sac_checkpoint(checkpoint)

        return checkpoint["episode"]

    def _load_base_checkpoint(self, checkpoint: dict):
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
        self.value_function.load_state_dict(checkpoint["value_state_dict"])
        self.value_optimizer.load_state_dict(checkpoint["value_optimizer_state_dict"])
        self.training_step = checkpoint.get("training_step", 0)
        self.action_low = checkpoint["action_low"].to(self.device)
        self.action_high = checkpoint["action_high"].to(self.device)

    def _load_encoder_checkpoint(self, checkpoint: dict):
        self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        self.encoder_optimizer.load_state_dict(checkpoint["encoder_optimizer_state_dict"])

    def _load_sac_checkpoint(self, checkpoint: dict):
        self.algorithm.q1.load_state_dict(checkpoint["q1_state_dict"])
        self.algorithm.q2.load_state_dict(checkpoint["q2_state_dict"])
        self.algorithm.target_q1.load_state_dict(checkpoint["target_q1_state_dict"])
        self.algorithm.target_q2.load_state_dict(checkpoint["target_q2_state_dict"])

        if "q1_optimizer_state_dict" in checkpoint:
            self.algorithm.q1_optimizer.load_state_dict(checkpoint["q1_optimizer_state_dict"])
            self.algorithm.q2_optimizer.load_state_dict(checkpoint["q2_optimizer_state_dict"])
