import torch
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal

from src.cqn.critic import C2FCritic
from src.cqn.encoder import MultiViewCNNEncoder
from src.cqn.networks import RandomShiftsAug


class TruncatedNormal(pyd.Normal):
    """Truncated normal distribution for action sampling."""

    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        """Clamp values to valid range using straight-through estimator."""
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        """Sample from truncated normal distribution."""
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


def _create_distribution(mean_action, stddev):
    """Create truncated normal distribution for action sampling."""
    stddev_tensor = torch.ones_like(mean_action) * stddev
    return TruncatedNormal(mean_action, stddev_tensor)


def _random_action(action):
    """Generate random action for exploration."""
    action.uniform_(-1.0, 1.0)
    return action


class ActionSelector:
    """Handles action selection logic for the agent."""

    def __init__(self, stddev_schedule, num_expl_steps):
        self.stddev_schedule = stddev_schedule
        self.num_expl_steps = num_expl_steps

    def select_action(self, mean_action, step, eval_mode):
        """Select action based on current policy and exploration strategy."""
        stddev = self._compute_stddev()
        distribution = _create_distribution(mean_action, stddev)

        if eval_mode:
            return distribution.mean

        action = distribution.sample(clip=None)

        if self._should_explore_randomly(step):
            action = _random_action(action)

        return action

    def _compute_stddev(self):
        """Compute standard deviation for action noise."""
        try:
            return float(self.stddev_schedule)
        except (ValueError, TypeError):
            return self.stddev_schedule

    def _should_explore_randomly(self, step):
        """Check if random exploration should be used."""
        return step < self.num_expl_steps


class DataAugmentation:
    """Handles data augmentation for observations."""

    def __init__(self, is_multiview, pad=4):
        self.is_multiview = is_multiview
        self.aug = RandomShiftsAug(pad=pad)

    def augment(self, rgb_obs):
        """Apply augmentation to RGB observations."""
        if self.is_multiview:
            return self._augment_multiview(rgb_obs)
        return self._augment_single_view(rgb_obs)

    def _augment_multiview(self, rgb_obs):
        """Apply augmentation to multi-view observations."""
        num_views = rgb_obs.shape[1]
        augmented_views = [
            self.aug(rgb_obs[:, v]) for v in range(num_views)
        ]
        return torch.stack(augmented_views, dim=1)

    def _augment_single_view(self, rgb_obs):
        """Apply augmentation to single-view observations."""
        return self.aug(rgb_obs)


class BatchProcessor:
    """Processes and prepares batches for training."""

    def __init__(self, device, data_augmentation, encoder):
        self.device = device
        self.data_augmentation = data_augmentation
        self.encoder = encoder

    def process_batch(self, batch):
        """Process raw batch into training-ready tensors."""
        tensors = self._convert_to_tensors(batch)
        rgb_obs, next_rgb_obs = self._prepare_rgb_observations(tensors)

        return {
            'rgb_obs': rgb_obs,
            'low_dim_obs': tensors[1],
            'action': tensors[2],
            'reward': tensors[3],
            'discount': tensors[4],
            'next_rgb_obs': next_rgb_obs,
            'next_low_dim_obs': tensors[6],
            'demos': tensors[7],
        }

    def _convert_to_tensors(self, batch):
        """Convert batch items to tensors on device."""
        return tuple(torch.as_tensor(x, device=self.device) for x in batch)

    def _prepare_rgb_observations(self, tensors):
        """Prepare and augment RGB observations."""
        rgb_obs = tensors[0].float()
        next_rgb_obs = tensors[5].float()

        rgb_obs = self.data_augmentation.augment(rgb_obs)
        next_rgb_obs = self.data_augmentation.augment(next_rgb_obs)

        rgb_obs = self.encoder(rgb_obs)
        with torch.no_grad():
            next_rgb_obs = self.encoder(next_rgb_obs)
        rgb_obs = rgb_obs.detach().requires_grad_(True)
        return rgb_obs, next_rgb_obs


class CriticUpdater:
    """Handles critic network updates."""

    def __init__(self, critic, critic_target, bc_lambda, bc_margin,
                 critic_lambda, use_logger):
        self.critic = critic
        self.critic_target = critic_target
        self.bc_lambda = bc_lambda
        self.bc_margin = bc_margin
        self.critic_lambda = critic_lambda
        self.use_logger = use_logger

    def compute_loss(self, batch_data):
        """Compute total critic loss."""
        metrics = {}

        target_q_probs = self._compute_target_distribution(batch_data, metrics)
        q_distributions = self._compute_current_distributions(batch_data)

        critic_loss = self._compute_critic_loss(q_distributions, target_q_probs, metrics)

        if self.bc_lambda > 0.0:
            bc_loss = self._compute_behavioral_cloning_loss(
                q_distributions, batch_data['demos'], metrics
            )
            critic_loss = critic_loss + bc_loss

        return critic_loss, metrics

    def _compute_target_distribution(self, batch_data, metrics):
        """Compute target Q-value distribution."""
        with torch.no_grad():
            next_action, action_metrics = self.critic.get_action(
                batch_data['next_rgb_obs'],
                batch_data['next_low_dim_obs']
            )

            target_q_probs = self.critic_target.compute_target_q_dist(
                batch_data['next_rgb_obs'],
                batch_data['next_low_dim_obs'],
                next_action,
                batch_data['reward'],
                batch_data['discount']
            )

            if self.use_logger:
                metrics.update(action_metrics)

        return target_q_probs

    def _compute_current_distributions(self, batch_data):
        """Compute current Q-value distributions."""
        q_probs, q_probs_a, log_q_probs, log_q_probs_a = self.critic(
            batch_data['rgb_obs'],
            batch_data['low_dim_obs'],
            batch_data['action']
        )

        return {
            'q_probs': q_probs,
            'q_probs_a': q_probs_a,
            'log_q_probs': log_q_probs,
            'log_q_probs_a': log_q_probs_a,
        }

    def _compute_critic_loss(self, q_distributions, target_q_probs, metrics):
        """Compute base critic loss."""
        q_critic_loss = -torch.sum(
            target_q_probs * q_distributions['log_q_probs_a'],
            dim=3
        ).mean()

        if self.use_logger:
            metrics["q_critic_loss"] = q_critic_loss.item()

        return self.critic_lambda * q_critic_loss

    def _compute_behavioral_cloning_loss(self, q_distributions, demos, metrics):
        """Compute behavioral cloning losses."""
        demos = demos.float().squeeze(1)

        if self.use_logger:
            metrics["ratio_of_demos"] = demos.mean().item()

        if torch.sum(demos) == 0:
            return 0.0

        bc_loss = self._compute_fosd_loss(q_distributions, demos, metrics)

        if self.bc_margin > 0:
            margin_loss = self._compute_margin_loss(q_distributions, demos, metrics)
            bc_loss = bc_loss + margin_loss

        return self.bc_lambda * bc_loss

    def _compute_fosd_loss(self, q_distributions, demos, metrics):
        """Compute first-order stochastic dominance loss."""
        q_probs_cdf = torch.cumsum(q_distributions['q_probs'], dim=-1)
        q_probs_a_cdf = torch.cumsum(q_distributions['q_probs_a'], dim=-1)

        fosd_loss = (
            (q_probs_a_cdf.unsqueeze(-2) - q_probs_cdf)
            .clamp(min=0)
            .sum(-1)
            .mean([-1, -2, -3])
        )

        fosd_loss = (fosd_loss * demos).sum() / demos.sum()

        if self.use_logger:
            metrics["bc_fosd_loss"] = fosd_loss.item()

        return fosd_loss

    def _compute_margin_loss(self, q_distributions, demos, metrics):
        """Compute margin-based loss for behavioral cloning."""
        qs = (
                q_distributions['q_probs'] *
                self.critic.support.expand_as(q_distributions['q_probs'])
        ).sum(-1)

        qs_a = (
                q_distributions['q_probs_a'] *
                self.critic.support.expand_as(q_distributions['q_probs_a'])
        ).sum(-1)

        margin_loss = torch.clamp(
            self.bc_margin - (qs_a.unsqueeze(-1) - qs),
            min=0
        ).mean([-1, -2, -3])

        margin_loss = (margin_loss * demos).sum() / demos.sum()

        if self.use_logger:
            metrics["bc_margin_loss"] = margin_loss.item()

        return margin_loss


class NetworkUpdater:
    """Handles network parameter updates."""

    def __init__(self, encoder_opt, critic_opt, critic_target_tau):
        self.encoder_opt = encoder_opt
        self.critic_opt = critic_opt
        self.critic_target_tau = critic_target_tau

    def update_networks(self, loss):
        """Update network parameters using computed loss."""
        self._zero_gradients()
        loss.backward()
        self._step_optimizers()

    def _zero_gradients(self):
        """Zero out gradients for all optimizers."""
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)

    def _step_optimizers(self):
        """Step all optimizers."""
        self.critic_opt.step()
        self.encoder_opt.step()

    def soft_update_target(self, critic, critic_target):
        """Soft update target network parameters."""
        for param, target_param in zip(critic.parameters(), critic_target.parameters()):
            target_param.data.copy_(
                self.critic_target_tau * param.data +
                (1 - self.critic_target_tau) * target_param.data
            )


class CQNAgent:
    """Coarse-to-Fine Q-Network agent for continuous control."""

    def __init__(
            self,
            rgb_obs_shape,
            low_dim_obs_shape,
            action_shape,
            device,
            lr,
            feature_dim,
            hidden_dim,
            levels,
            bins,
            atoms,
            v_min,
            v_max,
            bc_lambda,
            bc_margin,
            critic_lambda,
            critic_target_tau,
            weight_decay,
            num_expl_steps,
            update_every_steps,
            stddev_schedule,
            use_logger,
    ):
        self.device = device
        self.update_every_steps = update_every_steps
        self.use_logger = use_logger

        self._initialize_networks(
            rgb_obs_shape, low_dim_obs_shape, action_shape,
            feature_dim, hidden_dim, levels, bins, atoms, v_min, v_max
        )

        self._initialize_optimizers(lr, weight_decay)
        self._initialize_components(
            bc_lambda, bc_margin, critic_lambda,
            critic_target_tau, num_expl_steps, stddev_schedule
        )

        self.train()
        self.critic_target.eval()

    def _initialize_networks(self, rgb_obs_shape, low_dim_obs_shape, action_shape,
                             feature_dim, hidden_dim, levels, bins, atoms, v_min, v_max):
        """Initialize encoder and critic networks."""
        self.encoder = MultiViewCNNEncoder(rgb_obs_shape).to(self.device)
        self.is_multiview = len(rgb_obs_shape) == 4

        self.critic = C2FCritic(
            action_shape,
            self.encoder.repr_dim,
            low_dim_obs_shape,
            feature_dim,
            hidden_dim,
            levels,
            bins,
            atoms,
            v_min,
            v_max,
        ).to(self.device)

        self.critic_target = C2FCritic(
            action_shape,
            self.encoder.repr_dim,
            low_dim_obs_shape,
            feature_dim,
            hidden_dim,
            levels,
            bins,
            atoms,
            v_min,
            v_max,
        ).to(self.device)

        self.critic_target.load_state_dict(self.critic.state_dict())

    def _initialize_optimizers(self, lr, weight_decay):
        """Initialize optimizers for encoder and critic."""
        self.encoder_opt = torch.optim.AdamW(
            self.encoder.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        self.critic_opt = torch.optim.AdamW(
            self.critic.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    def _initialize_components(self, bc_lambda, bc_margin,
                               critic_lambda, critic_target_tau, num_expl_steps,
                               stddev_schedule):
        """Initialize agent components."""
        self.action_selector = ActionSelector(stddev_schedule, num_expl_steps)
        self.data_augmentation = DataAugmentation(self.is_multiview)
        self.batch_processor = BatchProcessor(
            self.device,
            self.data_augmentation,
            self.encoder
        )
        self.critic_updater = CriticUpdater(
            self.critic,
            self.critic_target,
            bc_lambda,
            bc_margin,
            critic_lambda,
            self.use_logger
        )
        self.network_updater = NetworkUpdater(
            self.encoder_opt,
            self.critic_opt,
            critic_target_tau
        )

    def train(self, training=True):
        """Set training mode for all networks."""
        self.training = training
        self.encoder.train(training)
        self.critic.train(training)

    def eval(self):
        """Set evaluation mode for all networks."""
        self.train(False)

    def act(self, rgb_obs, low_dim_obs, step, eval_mode):
        """Select action for given observation."""
        rgb_obs_tensor = self._prepare_observation(rgb_obs)
        low_dim_obs_tensor = self._prepare_observation(low_dim_obs)

        encoded_obs = self.encoder(rgb_obs_tensor)
        mean_action, _ = self.critic_target.get_action(encoded_obs, low_dim_obs_tensor)

        action = self.action_selector.select_action(mean_action, step, eval_mode)
        action = self.critic.encode_decode_action(action)

        return action.cpu().numpy()[0]

    def _prepare_observation(self, obs):
        """Prepare observation for network input."""
        return torch.as_tensor(obs, device=self.device).unsqueeze(0)

    def update(self, replay_iter, step):
        """Update agent with batch from replay buffer."""
        if not self._should_update(step):
            return {}

        batch = next(replay_iter)
        batch_data = self.batch_processor.process_batch(batch)

        metrics = self._compute_metrics(batch_data)
        loss, update_metrics = self.critic_updater.compute_loss(batch_data)
        metrics.update(update_metrics)

        self.network_updater.update_networks(loss)
        self.network_updater.soft_update_target(self.critic, self.critic_target)

        return metrics

    def _should_update(self, step):
        """Check if agent should be updated at this step."""
        return step % self.update_every_steps == 0

    def _compute_metrics(self, batch_data):
        """Compute additional metrics for logging."""
        metrics = {}
        if self.use_logger:
            metrics["batch_reward"] = batch_data['reward'].mean().item()
        return metrics
