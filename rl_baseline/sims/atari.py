from gym.wrappers import TimeLimit
from gym.envs.atari import AtariEnv

from rl_baseline.registry import sim_registry
from rl_baseline.core import Spec, EnvSim


class AtariSim(EnvSim):
    def __init__(self, spec):
        self._spec = spec
        super(AtariSim, self).__init__(self._spec.make())

    @staticmethod
    def get_state(wrapped_env):
        assert isinstance(wrapped_env, TimeLimit) and isinstance(wrapped_env.unwrapped, AtariEnv), 'Only accepts a `TimeLimit`-wrapped AtariEnv.'
        # Returns states from `TimeLimit` and AtariEnv
        return (wrapped_env._elapsed_steps, wrapped_env.unwrapped.clone_full_state())

    def set_state(self, state):
        elapsed_steps, atari_state = state
        # Set AtariEnv state
        self.env.unwrapped.restore_full_state(atari_state)
        self.env.unwrapped.steps_beyond_done = None
        # Set TimeLimit state
        self.env._elapsed_steps = elapsed_steps
