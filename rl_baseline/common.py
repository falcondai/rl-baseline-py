from six.moves import xrange

import logging

import numpy as np

from rl_baseline.util import log_format
from rl_baseline.core import Policy
from rl_baseline.registry import model_registry


logging.basicConfig(format=log_format)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@model_registry.register('random')
class RandomPolicy(Policy):
    '''A simple and useful baseline policy.'''
    def __init__(self, ob_space, ac_space):
        self.ac_space = ac_space

    def act(self, ob):
        return self.ac_space.sample()


def evaluate_policy(env, model, n_episodes, render):
    assert isinstance(model, Policy), '`model` must be an instance of `Policy`.'

    lens, rets = [], []
    for episode in xrange(n_episodes):
        ob = env.reset()
        done, length, ret = False, 0, 0
        while not done:
            ac = model.act(ob)
            ob, r, done, extra = env.step(ac)
            if render:
                env.render()
            ret += r
            length += 1
        logger.info('Episode %i return %g length %i', episode, ret, length)
        lens.append(length)
        rets.append(ret)
    logger.info('Total %i episodes', n_episodes)
    logger.info('Episode return mean/max/min/median %g/%g/%g/%g', np.mean(rets), np.max(rets), np.min(rets), np.median(rets))
    logger.info('Episode length mean/max/min/median %g/%g/%g/%g', np.mean(lens), np.max(lens), np.min(lens), np.median(lens))

    return rets, lens

# TODO train_policy
