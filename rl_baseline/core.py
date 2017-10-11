import argparse

from gym import Env, Wrapper
from gym.envs.registration import EnvSpec


class Spec:
    '''Store the entry point and parameters for items on the reigistry.'''
    def __init__(self, id, entry_point, **kwargs):
        self.id = id
        self.entry_point = entry_point
        self.kwargs = kwargs

    def make(self):
        return self.entry_point(**self.kwargs)


class GymEnvSpecWrapper(Spec):
    '''Wraps an existing `gym.envs.registration.EnvSpec` object to accommodate the custom `Env` instantiation behaviors in gym.'''
    def __init__(self, spec):
        '''
        Args:
            spec : `EnvSpec`
                An existing environment specification.
        '''
        assert isinstance(spec, EnvSpec), 'This is designed to wrap existing `EnvSpec` from `gym` only.'
        self._spec = spec
        self.id = self._spec.id
        super(GymEnvSpecWrapper, self).__init__(self.id, spec._entry_point)

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


class Policy:
    '''The abstraction of an agent. The main purpose of this class is to provide an API for evaluation.'''
    def __init__(self, *args, **kwargs):
        super(Policy, self).__init__(*args, **kwargs)

    def act(self, ob):
        '''Returns an action given the observation'''
        raise NotImplementedError


class StochasticPolicy(Policy):
    def __init__(self, *args, **kwargs):
        super(StochasticPolicy, self).__init__(*args, **kwargs)

    def pi(self, ob):
        # TODO return a function for non-discrete action spaces
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
    '''A simulator that needs to extract/estimate state from an environment and can set its state.'''
    def __init__(self, *args, **kwargs):
        super(Simulator, self).__init__(*args, **kwargs)

    def set_state(state):
        '''Sets the simulator state.'''
        raise NotImplementedError

    @staticmethod
    def get_state(env):
        '''Extracts and returns the state from an environment.'''
        raise NotImplementedError


class EnvSim(Simulator):
    '''A simulator that consists of an enviroment.'''
    def __init__(self, env, *args, **kwargs):
        self.env = env
        super(EnvSim, self).__init__(*args, **kwargs)

    def _step(self, ac):
        return self.env._step(ac)

    def _reset(self):
        return self.env._reset()

    def _render(self, mode='human', close=False):
        return self.env._render(mode, close)

    def _seed(self, seed=None):
        return self.env._seed(seed)


class Parsable:
    '''Inherits and implements `add_args` method in subclasses to add all the arguments to a CLI parser. The argument defaults should be defined in the `add_args` function instead of `__init__`.'''
    @classmethod
    def build_parser(kls, prefix):
        '''Creates a parser with its arguments'''
        parser = argparse.ArgumentParser()
        kls.add_args(parser, prefix)
        return parser

    @classmethod
    def add_args(kls, parser, prefix):
        # This is an example
        # parser.add_argument(kls.prefix_arg_name('exploration', prefix), dest='exploration_type', choices=['softmax', 'epsilon'], help='Type of exploration strategy.')
        raise NotImplementedError

    @staticmethod
    def prefix_arg_name(arg_name, prefix):
        if prefix != '':
            return '--%s-%s' % (prefix, arg_name)
        return '--%s' % arg_name
