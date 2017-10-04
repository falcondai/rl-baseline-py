from gym import Env, Wrapper
from gym.envs.registration import EnvSpec

class GymEnvSpecWrapper(EnvSpec):
    '''Wraps an existing `gym.envs.registration.EnvSpec` object to accommodate the custom `Env` instantiation behaviors in gym.'''
    def __init__(self, spec):
        '''Args:
        spec : `EnvSpec`. An existing environment specification.
        '''
        self._spec = spec
        self.id = self._spec.id

    def make(self):
        # Wrap TimeLimit over environment if needed
        spec = self._spec
        env = spec.make()
        if (env.spec.timestep_limit is not None) and not spec.tags.get('vnc'):
            from gym.wrappers.time_limit import TimeLimit
            env = TimeLimit(env,
                            max_episode_steps=env.spec.max_episode_steps,
                            max_episode_seconds=env.spec.max_episode_seconds)
        return env

class StochasticPolicy:
    def __init__(self, *args, **kwargs):
        super(StochasticPolicy, self).__init__(*args, **kwargs)

    def pi(self, ob):
        raise NotImplementedError

    def sample_ac(self, ac_prob):
        '''Sample an action based on the action probability `ac_prob`.'''
        raise NotImplementedError


class StateValue:
    def __init__(self, *args, **kwargs):
        super(StateValue, self).__init__(*args, **kwargs)

    def va(self, ob):
        raise NotImplementedError


class ActionValue:
    def __init__(self, *args, **kwargs):
        super(ActionValue, self).__init__(*args, **kwargs)

    def q(self, ob):
        raise NotImplementedError


class Simulator(Env):
    def load_state():
        raise NotImplementedError

    def save_state():
        raise NotImplementedError
