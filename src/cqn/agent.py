import torch

import utils
from src.cqn.encoder import MultiViewCNNEncoder
from src.cqn.critic import C2FCritic
from src.cqn.networks import RandomShiftsAug


class CQNAgent:
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
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_logger = use_logger
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.bc_lambda = bc_lambda
        self.bc_margin = bc_margin
        self.critic_lambda = critic_lambda
        self.encoder = MultiViewCNNEncoder(rgb_obs_shape).to(device)

        # Check if multi-view or single-view
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
        ).to(device)
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
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.encoder_opt = torch.optim.AdamW(
            self.encoder.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.critic_opt = torch.optim.AdamW(
            self.critic.parameters(), lr=lr, weight_decay=weight_decay
        )

        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.eval()

        print(self.encoder)
        print(self.critic)

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.critic.train(training)

    def act(self, rgb_obs, low_dim_obs, step, eval_mode):
        rgb_obs = torch.as_tensor(rgb_obs, device=self.device).unsqueeze(0)
        low_dim_obs = torch.as_tensor(low_dim_obs, device=self.device).unsqueeze(0)
        rgb_obs = self.encoder(rgb_obs)
        stddev = utils.schedule(self.stddev_schedule, step)
        action, _ = self.critic_target.get_action(
            rgb_obs, low_dim_obs
        )
        stddev = torch.ones_like(action) * stddev
        dist = utils.TruncatedNormal(action, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        action = self.critic.encode_decode_action(action)
        return action.cpu().numpy()[0]

    def update_critic(
            self,
            rgb_obs,
            low_dim_obs,
            action,
            reward,
            discount,
            next_rgb_obs,
            next_low_dim_obs,
            demos,
    ):
        metrics = dict()

        with torch.no_grad():
            next_action, mets = self.critic.get_action(next_rgb_obs, next_low_dim_obs)
            target_q_probs_a = self.critic_target.compute_target_q_dist(
                next_rgb_obs, next_low_dim_obs, next_action, reward, discount
            )
            if self.use_logger:
                metrics.update(**mets)

        q_probs, q_probs_a, log_q_probs, log_q_probs_a = self.critic(
            rgb_obs, low_dim_obs, action
        )
        q_critic_loss = -torch.sum(target_q_probs_a * log_q_probs_a, 3).mean()
        critic_loss = self.critic_lambda * q_critic_loss

        if self.use_logger:
            metrics["q_critic_loss"] = q_critic_loss.item()

        if self.bc_lambda > 0.0:
            demos = demos.float().squeeze(1)
            if self.use_logger:
                metrics["ratio_of_demos"] = demos.mean().item()

            if torch.sum(demos) > 0:
                q_probs_cdf = torch.cumsum(q_probs, -1)
                q_probs_a_cdf = torch.cumsum(q_probs_a, -1)
                bc_fosd_loss = (
                    (q_probs_a_cdf.unsqueeze(-2) - q_probs_cdf)
                    .clamp(min=0)
                    .sum(-1)
                    .mean([-1, -2, -3])
                )
                bc_fosd_loss = (bc_fosd_loss * demos).sum() / demos.sum()
                critic_loss = critic_loss + self.bc_lambda * bc_fosd_loss
                if self.use_logger:
                    metrics["bc_fosd_loss"] = bc_fosd_loss.item()

                if self.bc_margin > 0:
                    qs = (q_probs * self.critic.support.expand_as(q_probs)).sum(-1)
                    qs_a = (q_probs_a * self.critic.support.expand_as(q_probs_a)).sum(
                        -1
                    )
                    margin_loss = torch.clamp(
                        self.bc_margin - (qs_a.unsqueeze(-1) - qs), min=0
                    ).mean([-1, -2, -3])
                    margin_loss = (margin_loss * demos).sum() / demos.sum()
                    critic_loss = critic_loss + self.bc_lambda * margin_loss
                    if self.use_logger:
                        metrics["bc_margin_loss"] = margin_loss.item()

        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        (
            rgb_obs,
            low_dim_obs,
            action,
            reward,
            discount,
            next_rgb_obs,
            next_low_dim_obs,
            demos,
        ) = utils.to_torch(batch, self.device)

        rgb_obs = rgb_obs.float()
        next_rgb_obs = next_rgb_obs.float()

        # Apply augmentation based on whether it's multi-view or single-view
        if self.is_multiview:
            # Multi-view: [B, num_views, C, H, W]
            rgb_obs = torch.stack(
                [self.aug(rgb_obs[:, v]) for v in range(rgb_obs.shape[1])], 1
            )
            next_rgb_obs = torch.stack(
                [self.aug(next_rgb_obs[:, v]) for v in range(next_rgb_obs.shape[1])], 1
            )
        else:
            # Single-view: [B, C, H, W]
            rgb_obs = self.aug(rgb_obs)
            next_rgb_obs = self.aug(next_rgb_obs)

        rgb_obs = self.encoder(rgb_obs)
        with torch.no_grad():
            next_rgb_obs = self.encoder(next_rgb_obs)

        if self.use_logger:
            metrics["batch_reward"] = reward.mean().item()

        metrics.update(
            self.update_critic(
                rgb_obs,
                low_dim_obs,
                action,
                reward,
                discount,
                next_rgb_obs,
                next_low_dim_obs,
                demos,
            )
        )

        utils.soft_update_params(
            self.critic, self.critic_target, self.critic_target_tau
        )

        return metrics