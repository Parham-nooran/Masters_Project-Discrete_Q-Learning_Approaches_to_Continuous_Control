"""Factory for creating environments from different sources."""
import ogbench

from dm_control import suite

def create_ogbench_env(task_name, seed=0):
    """Create OGBench environment.

    Args:
        task_name: Name of OGBench task (e.g., 'antmaze-large-navigate-v0')
        seed: Random seed

    Returns:
        OGBench environment
    """
    env = ogbench.make_env_and_datasets(task_name, env_only=True)
    env = OGBenchWrapper(env, seed)
    return env


def create_dmcontrol_env(domain_name, task_name, seed=0):
    """Create dm_control environment.

    Args:
        domain_name: Domain name (e.g., 'walker')
        task_name: Task name (e.g., 'walk')
        seed: Random seed

    Returns:
        dm_control environment
    """
    env = suite.load(domain_name=domain_name, task_name=task_name, task_kwargs={'random': seed})
    return env


class OGBenchWrapper:
    """Wrapper to make OGBench environments compatible with dm_control interface."""

    def __init__(self, env, seed=0):
        self.env = env
        self.seed = seed
        self._step_count = 0
        self._max_episode_steps = 1000

    def reset(self):
        """Reset environment."""
        obs, info = self.env.reset(seed=self.seed)
        self._step_count = 0

        class TimeStep:
            def __init__(self, observation):
                self.observation = observation

        return TimeStep(obs)

    def step(self, action):
        """Take environment step."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1
        done = terminated or truncated

        class TimeStep:
            def __init__(self, observation, reward, done):
                self.observation = observation
                self.reward = reward
                self._done = done

            def last(self):
                return self._done

        return TimeStep(obs, reward, done)

    def action_spec(self):
        """Get action specification."""

        class ActionSpec:
            def __init__(self, space):
                self.minimum = space.low
                self.maximum = space.high
                self.shape = space.shape

        return ActionSpec(self.env.action_space)

    def observation_spec(self):
        """Get observation specification."""
        obs_space = self.env.observation_space

        if hasattr(obs_space, 'shape'):
            return {'observations': type('obj', (), {'shape': obs_space.shape})}
        else:
            return {k: type('obj', (), {'shape': v.shape})
                    for k, v in obs_space.spaces.items()}