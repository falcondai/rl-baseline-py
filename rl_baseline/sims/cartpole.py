from gym.wrappers import TimeLimit
from gym.envs.classic_control import CartPoleEnv, MountainCarEnv

from rl_baseline.registry import sim_registry
from rl_baseline.core import Spec, EnvSim


class CartPoleSim(EnvSim):
    def __init__(self, spec):
        self._spec = spec
        super(CartPoleSim, self).__init__(self._spec.make())

    @staticmethod
    def get_state(wrapped_cartpole_env):
        assert isinstance(wrapped_cartpole_env, TimeLimit) and isinstance(wrapped_cartpole_env.unwrapped, CartPoleEnv), 'Only accepts a `TimeLimit`-wrapped CartPoleEnv.'
        # Returns states from `TimeLimit` and CartPoleEnv
        return (wrapped_cartpole_env._elapsed_steps, wrapped_cartpole_env.unwrapped.state)

    def set_state(self, state):
        elapsed_steps, cartpole_state = state
        # Set CartPoleEnv state
        self.env.unwrapped.state = cartpole_state
        self.env.unwrapped.steps_beyond_done = None
        # Set TimeLimit state
        self.env._elapsed_steps = elapsed_steps

class MountainCarSim(EnvSim):
    def __init__(self, spec):
        self._spec = spec
        super(MountainCarSim, self).__init__(self._spec.make())

    @staticmethod
    def get_state(wrapped_moutaincar_env):
        assert isinstance(wrapped_moutaincar_env, TimeLimit) and isinstance(wrapped_moutaincar_env.unwrapped, MountainCarEnv), 'Only accepts a `TimeLimit`-wrapped MountainCarEnv.'
        # Returns states from `TimeLimit` and MountainCarEnv
        return (wrapped_moutaincar_env._elapsed_steps, wrapped_moutaincar_env.unwrapped.state)

    def set_state(self, state):
        elapsed_steps, mountaincar_state = state
        # Set CartPoleEnv state
        self.env.unwrapped.state = mountaincar_state
        self.env.unwrapped.steps_beyond_done = None
        # Set TimeLimit state
        self.env._elapsed_steps = elapsed_steps
