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

    def __init__(self, args, obs_shape: tuple, action_spec: dict):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.training_step = 0
        self.last_obs = None

        self._setup_action_space(action_spec)
        self._setup_networks(args, obs_shape)
        self._setup_optimizers(args)
        self.replay_buffer = self._create_replay_buffer(args)
        self.algorithm = self._create_algorithm(args)

    def _setup_action_space(self, action_spec: dict):
        self.action_spec = action_spec
        self.action_dim = len(action_spec["low"])

        self.action_low = torch.tensor(
            action_spec["low"], device=self.device, dtype=torch.float32
        )
        self.action_high = torch.tensor(
            action_spec["high"], device=self.device, dtype=torch.float32
        )

    def _setup_networks(self, args, obs_shape: tuple):
        self.encoder, self.encoder_output_size = self._create_encoder(args, obs_shape)
        self.policy = self._create_policy(args)
        self.value_function = self._create_value_function(args)

    def _setup_optimizers(self, args):
        self.policy_optimizer = optim.Adam(
            self.policy.parameters(), lr=args.learning_rate
        )
        self.value_optimizer = optim.Adam(
            self.value_function.parameters(), lr=args.learning_rate
        )
        self.encoder_optimizer = self._create_encoder_optimizer(args)

    def _create_encoder(
        self, args, obs_shape: tuple
    ) -> Tuple[Optional[VisionEncoder], int]:
        if args.use_pixels:
            encoder = VisionEncoder(args, args.num_pixels).to(self.device)
            return encoder, args.layer_size_bottleneck
        return None, int(np.prod(obs_shape))

    def _create_encoder_optimizer(self, args) -> Optional[optim.Adam]:
        if self.encoder:
            return optim.Adam(self.encoder.parameters(), lr=args.learning_rate)
        return None

    def _create_policy(self, args) -> BernoulliPolicy:
        return BernoulliPolicy(
            self.encoder_output_size, self.action_dim, args.layer_size_network
        ).to(self.device)

    def _create_value_function(self, args) -> LayerNormMLP:
        from src.common.networks import LayerNormMLP

        layer_sizes = [self.encoder_output_size] + args.layer_size_network + [1]
        return LayerNormMLP(layer_sizes, activate_final=False).to(self.device)

    def _create_replay_buffer(self, args) -> PrioritizedReplayBuffer:
        buffer = PrioritizedReplayBuffer(
            capacity=args.max_replay_size,
            alpha=args.priority_exponent,
            beta=args.importance_sampling_exponent,
            n_step=args.adder_n_step,
            discount=args.discount,
        )
        buffer.device = self.device
        return buffer

    def _create_algorithm(self, args) -> Base:
        algorithm_type = args.algorithm.lower()

        if algorithm_type == "ppo":
            return PPO(
                clip_ratio=args.ppo_clip_ratio,
                value_coef=args.ppo_value_coef,
            )
        elif algorithm_type == "sac":
            sac = SAC(
                alpha=args.sac_alpha,
                tau=args.sac_tau,
                learning_rate=args.learning_rate,
            )
            sac.initialize_q_networks(
                self.encoder_output_size + self.action_dim, self.device
            )
            return sac
        elif algorithm_type == "mpo":
            return MPO(
                epsilon=args.mpo_epsilon,
                epsilon_penalty=args.mpo_epsilon_penalty,
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_type}")

    def _encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation with numerical stability checks."""
        if self.encoder:
            encoded = self.encoder(obs)
        else:
            encoded = obs.flatten(1)

        # Check for NaN/Inf
        if torch.isnan(encoded).any() or torch.isinf(encoded).any():
            print(f"Warning: NaN/Inf in encoded observation")
            encoded = torch.nan_to_num(encoded, nan=0.0, posinf=1.0, neginf=-1.0)

        return encoded

    def observe_first(self, obs: torch.Tensor):
        self.last_obs = obs

    def select_action(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        obs = self._prepare_observation(obs)

        with torch.no_grad():
            encoded_obs = self._encode_obs(obs)
            action, _ = self.policy.get_action(encoded_obs, deterministic)

        return action.squeeze(0)

    def _prepare_observation(self, obs: torch.Tensor) -> torch.Tensor:
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().to(self.device)

        if len(obs.shape) == 1 or (self.encoder and len(obs.shape) == 3):
            obs = obs.unsqueeze(0)

        return obs

    def observe(
        self, action: torch.Tensor, reward: float, next_obs: torch.Tensor, done: bool
    ):
        if self.last_obs is None:
            return

        binary_action = (action + 1.0) / 2.0

        self.replay_buffer.add(self.last_obs, binary_action, reward, next_obs, done)
        self.last_obs = next_obs.detach()

    def update(self) -> Dict[str, float]:
        batch = self._sample_batch()
        if batch is None:
            return {}

        components = get_batch_components(batch, self.device)
        obs, actions, rewards, next_obs, dones, discounts, weights, indices = components

        obs_encoded, next_obs_encoded = encode_observation(self.encoder, obs, next_obs)

        # CRITICAL FIX: Ensure actions are properly binary (0 or 1)
        # Clamp first to handle any out-of-range values from replay
        actions = torch.clamp(actions, 0.0, 1.0)
        # Round to nearest integer to ensure exact 0 or 1
        actions = torch.round(actions)

        # Normalize rewards for stability
        if rewards.std() > 1e-6:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

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

        # Check for NaN loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(
                f"Warning: NaN/Inf loss detected at step {self.training_step}, skipping update"
            )
            return {"loss_error": 1.0}

        self._perform_gradient_update(loss)
        self.algorithm.update_target_networks()
        self._update_replay_priorities(indices, td_errors)

        self.training_step += 1
        return metrics

    def _sample_batch(self) -> Optional[dict]:
        return check_and_sample_batch_from_replay_buffer(
            self.replay_buffer, self.args.min_replay_size, self.args.batch_size
        )

    def _perform_gradient_update(self, loss: torch.Tensor):
        optimizers = [self.policy_optimizer, self.value_optimizer]
        if self.encoder_optimizer:
            optimizers.append(self.encoder_optimizer)
        optimizers.extend(self.algorithm.get_optimizers())

        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

        loss.backward()

        if self.args.clip_gradients:
            self._clip_gradients()

        for optimizer in optimizers:
            optimizer.step()

    def _clip_gradients(self):
        clip_norm = self.args.clip_gradients_norm
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), clip_norm)
        torch.nn.utils.clip_grad_norm_(self.value_function.parameters(), clip_norm)
        if self.encoder:
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), clip_norm)

    def _update_replay_priorities(self, indices, td_errors):
        if td_errors is not None:
            priorities = td_errors.detach().cpu().numpy()
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
            "args": self.args,
        }

        if self.encoder:
            checkpoint["encoder_state_dict"] = self.encoder.state_dict()
            checkpoint["encoder_optimizer_state_dict"] = (
                self.encoder_optimizer.state_dict()
            )

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
        self.encoder_optimizer.load_state_dict(
            checkpoint["encoder_optimizer_state_dict"]
        )

    def _load_sac_checkpoint(self, checkpoint: dict):
        self.algorithm.q1.load_state_dict(checkpoint["q1_state_dict"])
        self.algorithm.q2.load_state_dict(checkpoint["q2_state_dict"])
        self.algorithm.target_q1.load_state_dict(checkpoint["target_q1_state_dict"])
        self.algorithm.target_q2.load_state_dict(checkpoint["target_q2_state_dict"])

        if "q1_optimizer_state_dict" in checkpoint:
            self.algorithm.q1_optimizer.load_state_dict(
                checkpoint["q1_optimizer_state_dict"]
            )
            self.algorithm.q2_optimizer.load_state_dict(
                checkpoint["q2_optimizer_state_dict"]
            )
