import argparse

from gym import Env, Wrapper
from gym.envs.registration import EnvSpec

from rl_baseline.util import logger, write_tb_event


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
        self.observation_space = env.observation_space
        self.action_space = env.action_space
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
        # We want to call the `add_args` of the current class, that s why we need classmethod here.
        kls.add_args(parser, prefix)
        return parser

    # XXX is this necessary? Can we use staticmethod instead?
    @classmethod
    def add_args(kls, parser, prefix):
        # This is an example in `DqnTrainer`
        # parser.add_argument(kls.prefix_arg_name('exploration', prefix), dest='exploration_type', choices=['softmax', 'epsilon'], help='Type of exploration strategy.')
        pass

    @staticmethod
    def prefix_arg_name(arg_name, prefix):
        if prefix != '':
            return '--%s-%s' % (prefix, arg_name)
        return '--%s' % arg_name


class Trainer:
    '''Abstraction for RL algorithms.'''
    @classmethod
    def scaffold(kls, env_id, model_id, simulator_id):
        '''High level interface to hydrate string registry keys into live models and environments and simulators for the learning algorithm, e.g., creating an extra target model for DQN. This should generate keyword arguments for `__init__`. Must be overridden in subclasses.'''
        from rl_baseline.registration import env_registry, model_registry, sim_registry
        kwargs = dict(
            env = env_registry[env_id].make(),
            model = model_registry[model_id],
        )
        return kwargs

    def __init__(self, env, model, optimizer, train_summary_writer=None, eval_summary_writer=None, eval_env=None, n_eval_episodes=0, gpu_id=None):
        # TODO change model to agent to avoid confusion
        assert (eval_env is not None) == (n_eval_episodes > 0), '`eval_env` is needed iff `n_eval_episodes` is greater than 0.'
        self.env = env
        self.model = model
        self.optimizer = optimizer
        self.eval_env = eval_env
        self.train_summary_writer = train_summary_writer
        self.eval_summary_writer = eval_summary_writer
        self.n_eval_episodes = n_eval_episodes
        # GPU utility
        self.gpu_id = gpu_id
        self.to_model_device = lambda t : t.cuda(self.gpu_id) if self.gpu_id is not None else t.cpu()
        # Logging
        self.episode = 0
        self.tick = 0
        self.step = 0
        self.cumulative_reward_per_tick = 0
        self.cumulative_reward_per_episode = 0
        self._running_episode_return = 0
        self._running_episode_length = 0

    def update_stats(self, reward, done):
        self._running_episode_return += reward
        self._running_episode_length += 1
        self.cumulative_reward_per_tick = (self.tick * self.cumulative_reward_per_tick + reward) / (self.tick + 1)
        self.tick += 1
        # Update episodic stats
        if done:
            self.cumulative_reward_per_episode = (self.episode * self.cumulative_reward_per_episode + self._running_episode_return) / (self.episode + 1)
            self.episode += 1
            # Report stats
            self.report_episode(self._running_episode_return, self._running_episode_length)
            self._running_episode_return = 0
            self._running_episode_length = 0

    def report_episode(self, episode_return, episode_length):
        '''Report statistics of an episode.'''
        logger.info('Episode %i length %i return %g', self.episode, episode_length, episode_return)
        if self.train_summary_writer is not None:
            write_tb_event(self.train_summary_writer, self.episode, {
                'episodic/length': episode_length,
                'episodic/return': episode_return,
            })

            write_tb_event(self.train_summary_writer, self.tick, {
                'metrics/episode_return': episode_return,
            })

    def report_step(self, n_eval_episodes):
        '''Report progress of learning.'''
        # Online metrics (online view)
        logger.info('Step %i cumulative return / tick %.2f cumulative return / episode %.2f', self.step, self.cumulative_reward_per_tick, self.cumulative_reward_per_episode)
        if self.train_summary_writer is not None:
            write_tb_event(self.train_summary_writer, self.tick, {
                'metrics/cumulative_return_per_tick': self.cumulative_reward_per_tick,
                'metrics/cumulative_reward_per_episode': self.cumulative_reward_per_episode,
            })
            write_tb_event(self.train_summary_writer, self.episode, {
                'episodic/cumulative_reward_per_episode': self.cumulative_reward_per_episode,
            })
        # Offline metrics (policy optimization view)
        if n_eval_episodes > 0:
            # Run a few evaluation runs
            rets, lens = self.evaluate(n_eval_episodes=n_eval_episodes)
            report_perf(rets, lens)
            if self.eval_summary_writer is not None:
                write_tb_event(self.eval_summary_writer, self.tick, {
                    'metrics/episode_return': np.mean(rets),
                })

    def evaluate(self, eval_env, n_eval_episodes, render=False):
        '''Evaluate the current model by running on a few episodes.'''
        assert eval_env is not None, 'Must use a separate environment `eval_env` to evaluate.'
        return evaluate_policy(env=eval_env, policy=self.model, n_episodes=n_eval_episodes, render=render, gpu_id=self.gpu_id)

    def train_for(self, max_ticks):
        raise NotImplementedError
