import torch
import random
import numpy as np


class CustomDiscreteFeedForwardActor:
    def __init__(
        self,
        policy_network,
        encoder=None,
        action_discretizer=None,
        epsilon=0.1,
        decouple=False,
    ):
        self.policy_network = policy_network
        self.encoder = encoder
        self.action_discretizer = action_discretizer
        self.epsilon = epsilon
        self.decouple = decouple
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_network.eval()
        if self.encoder:
            self.encoder.eval()

    def check_and_get_observation(self, observation):
        if isinstance(observation, np.ndarray):
            observation = (
                torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
            )
        elif isinstance(observation, torch.Tensor):
            if observation.device != self.device:
                observation = observation.to(self.device)
            if len(observation.shape) < 2 or (
                    self.encoder and len(observation.shape) == 3
            ):
                observation = observation.unsqueeze(0)
        return observation

    def select_action(self, observation):
        observation = self.check_and_get_observation(observation)

        with torch.no_grad():
            if self.encoder:
                encoded_obs = self.encoder(observation)
            else:
                encoded_obs = observation.flatten(1)

            q1, q2 = self.policy_network(encoded_obs)
            discrete_action = self.epsilon_greedy_action((q1, q2))

            if self.action_discretizer:
                continuous_action = self.action_discretizer.discrete_to_continuous(
                    discrete_action
                )[0]
                return continuous_action.detach()
            else:
                return discrete_action[0]

    def epsilon_greedy_action(self, q_values):
        q1, q2 = q_values
        q_combined = torch.max(q1, q2)
        batch_size = q_combined.shape[0]

        if self.decouple:
            random_mask = (
                torch.rand(batch_size, q_combined.shape[1], device=self.device)
                < self.epsilon
            )

            num_actions = q_combined.shape[2]
            random_actions = torch.randint(
                0, num_actions, (batch_size, q_combined.shape[1]), device=self.device
            )
            greedy_actions = q_combined.argmax(dim=2)
            actions = torch.where(random_mask, random_actions, greedy_actions)
            return actions
        else:
            if random.random() < self.epsilon:
                return torch.randint(
                    0, q_combined.shape[1], (batch_size,), device=self.device
                )
            else:
                return q_combined.argmax(dim=1)
