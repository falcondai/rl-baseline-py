from six.moves import xrange

import logging

import numpy as np
from gym import spaces
import torch
from torch import nn

from rl_baseline.util import report_perf, logger
from rl_baseline.core import Policy, Parsable
from rl_baseline.registration import model_registry


@model_registry.register('random')
class RandomPolicy(Policy, Parsable):
    '''A simple and useful baseline policy.'''
    def __init__(self, ob_space, ac_space):
        self.ac_space = ac_space

    def act(self, ob):
        return self.ac_space.sample()


def q_from_rollout(sim, state, action, rollout_policy, depth_bound=None, va=None, gamma=1):
    '''Sample a gamma-discounted return from a rollout, optionally truncated and bootstrapped with a value function.
    Args:
        rollout_policy : `Policy`
            The policy used to produce a rollout.
        depth_bound : int
            The bound on the length of each rollout.
        va : `StateValue`
            The bootstrapping value function to add to the return of a partial rollout.
        gamma : float
            The discount factor of gamma-discounted return. Gamma-discounted return G = \sum_{t=0} \gamma^t r_t.
    '''
    # Start simulator at the current state
    sim.reset()
    sim.set_state(state)
    # Execute the current action
    ob, r, done, extra = sim.step(action)
    total_return, t = r, 0
    coeff = gamma
    while not done and (depth_bound is None or t < depth_bound):
        ac = rollout_policy.act(ob)
        ob, r, done, extra = sim.step(ac)
        total_return += r * coeff
        t += 1
        coeff *= gamma
    # Bootstrap the return with the provided value function
    if not done and va is not None:
        total_return += coeff * va(ob)
    return total_return

@model_registry.register('exhaust')
class ExhaustivePolicy(Policy, Parsable):
    '''Bounded-depth exhaustive search.'''
    @classmethod
    def add_args(kls, parser, prefix):
        parser.add_argument(kls.prefix_arg_name('sim', prefix), dest='sim', help='Simulator to use.')
        parser.add_argument(kls.prefix_arg_name('bound', prefix), dest='search_bound', type=int, default=None, help='Max depth of each rollout. None for no bound.')
        parser.add_argument(kls.prefix_arg_name('n-rollouts', prefix), dest='max_rollouts', type=int, default=100, help='Number of rollouts for each tick.')

    def __init__(self, env, sim, rollout_policy, max_rollouts, search_bound):
        assert sim.action_space == env.action_space, 'The action spaces of `env` and `sim` have to be the same.'
        self.env = env
        self.sim = sim
        self.ac_space = env.action_space
        self.rollout_policy = rollout_policy
        self.max_rollouts = max_rollouts
        self.search_bound = search_bound

    def act(self, ob):
        st = self.sim.get_state(self.env)
        acs, avgs, n_samples = [], [], []
        # Use rollouts to find the best action
        for i in xrange(self.max_rollouts):
            # Take a random action
            ac = self.ac_space.sample()
            try:
                ac_idx = acs.index(ac)
            except ValueError:
                # The action `ac` is missing, append it to the list
                acs.append(ac)
                n_samples.append(0)
                avgs.append(0)
                ac_idx = len(acs) - 1
            # Rollouts and Monte-Carlo estimates
            q = q_from_rollout(self.sim, st, ac, rollout_policy=self.rollout_policy, depth_bound=self.search_bound, va=None)
            # Update statistics
            avgs[ac_idx] = (n_samples[ac_idx] * avgs[ac_idx] + q) / (n_samples[ac_idx] + 1)
            n_samples[ac_idx] += 1
        # Pick the best action
        max_ac_idx = np.argmax(avgs)
        return acs[max_ac_idx]


def evaluate_policy(env, policy, n_episodes, render, gpu_id=None, seed=None):
    assert isinstance(policy, Policy), '`policy` must be an instance of `Policy`.'
    # Policy is also a trainable model
    is_model = isinstance(policy, nn.Module)

    if seed is not None:
        logger.debug('Eval env is seeded with %i' % seed)
        env.seed(seed)
    lens, rets = [], []
    for episode in xrange(n_episodes):
        ob = env.reset()
        done, length, ret = False, 0, 0
        while not done:
            if is_model:
                ac = policy.act(ob, gpu_id=gpu_id)
            else:
                ac = policy.act(ob)
            ob, r, done, extra = env.step(ac)
            if render:
                env.render()
            ret += r
            length += 1
        logger.debug('Episode %i return %g length %i', episode, ret, length)
        lens.append(length)
        rets.append(ret)
    report_perf(rets, lens, log_level=logging.DEBUG)

    return rets, lens


def load_model(checkpoint_path, env, model_class, model_args_dict, gpu_id):
    if gpu_id is None:
        # Using CPU, return the deserialized storage on CPU
        map_location = lambda storage, loc: storage
    else:
        # Using GPU
        map_location = lambda storage, loc: storage.cuda(gpu_id)
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    logger.info('Loading model from %s', checkpoint_path)
    # Use the model arguments saved in checkpoint
    model_args_dict = model_args_dict or checkpoint['model_args']
    model_class = model_class or model_registry[checkpoint['model_id']]
    model = model_class(env.observation_space, env.action_space, **model_args_dict)
    model.load_state_dict(checkpoint['model'])
    return model, checkpoint
