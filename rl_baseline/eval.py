from six.moves import xrange

import logging
from operator import xor

import numpy as np
import torch

from rl_baseline.registration import env_registry, model_registry
from rl_baseline.util import log_format, fix_random_seeds

logging.basicConfig(format=log_format)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if __name__ == '__main__':
    import argparse, os

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--log-dir', help='Path to the log directory containing checkpoints.')
    parser.add_argument('--checkpoint', help='Path to a specific checkpoint.')
    parser.add_argument('-n', '--n-episodes', type=int, default=10)
    parser.add_argument('-e', '--environment', default='gym.CartPole-v0')
    parser.add_argument('-m', '--model', default='dqn.mlp')
    parser.add_argument('-r', '--render', action='store_true', help='Show the environment.')
    parser.add_argument('-s', '--seed', type=int, help='Random seed.')

    args = parser.parse_args()

    assert xor(bool(args.checkpoint), bool(args.log_dir)), 'Please only provide path to the log directory OR a specific checkpoint.'

    env = env_registry[args.environment].make()
    mod = model_registry[args.model](env.observation_space, env.action_space)

    # Fix random seed
    if args.seed is not None:
        fix_random_seeds(args.seed, env, torch, np)

    # Load checkpoint
    if args.log_dir is not None:
        checkpoint_path = os.path.join(args.log_dir, 'checkpoint_t10000.pt')
    else:
        checkpoint_path = args.checkpoint
    checkpoint = torch.load(checkpoint_path)
    mod.load_state_dict(checkpoint['model'])

    lengths, rets = [], []
    for episode in xrange(args.n_episodes):
        ob = env.reset()
        done, length, ret = False, 0, 0
        while not done:
            ac = mod.act(ob)
            ob, r, done, extra = env.step(ac)
            if args.render:
                env.render()
            ret += r
            length += 1
        logger.info('Episode %i return %g length %i', episode, ret, length)
        lengths.append(length)
        rets.append(ret)
    logger.info('Total %i episodes average return %g average length %g', args.n_episodes, np.mean(rets), np.mean(lengths))
