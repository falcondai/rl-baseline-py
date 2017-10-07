from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging, itertools
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
from torch.nn import functional as f

import gym
gym.undo_logger_setup()

from registration import env_registry, optimizer_registry, model_registry, method_registry

from util import log_format, global_norm, get_cartpole_state, set_cartpole_state, copy_params

logging.basicConfig(format=log_format)
logger = logging.getLogger(__name__)

# Set the logging level
logger.setLevel(logging.DEBUG)
# Debug information
logger.debug('PyTorch version %s', torch.__version__)

if __name__ == '__main__':
    import time, argparse, os

    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--environment', default='gym.CartPole-v0', choices=env_registry.all().keys(), help='Environment id.')
    parser.add_argument('-m', '--model', default='a2c.linear', choices=model_registry.all().keys(), help='Model id.')
    # TODO add checkpoint save/restore
    parser.add_argument('-l', '--log-dir', default='/tmp/ail/t', help='Path to log directory.')
    parser.add_argument('--no-summary', dest='write_summary', action='store_false', help='Do not write summary protobuf for TensorBoard.')
    parser.add_argument('--episode-report-interval', type=int, default=1, help='Report every N-many episodes.')
    parser.add_argument('--step-report-interval', type=int, default=1, help='Report every N-many steps.')
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.05, help='Initial learning rate.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed.')
    parser.add_argument('-n', '--max-ticks', type=int, default=10**4, help='Maximum number of ticks to train.')
    parser.add_argument('-b', '--batch-size', type=int, default=20, help='Batch size.')
    parser.add_argument('-o', '--optimizer', default='SGD', choices=optimizer_registry.all().keys(), help='Optimizer to use.')
    parser.add_argument('--render', action='store_true', help='Show the environment.')
    # TODO add LR scheduler
    parser.add_argument('--lr-scheduler', default='none', help='Scheduler for learning rates.')
    # TODO add GPU support
    parser.add_argument('--gpu', dest='gpu_id', default=None, help='GPU id to use.')

    args = parser.parse_args()

    # Summary for TensorBoard monitoring
    if args.write_summary:
        try:
            # Tensorflow imports for writing summaries
            from tensorflow import summary
            logger.debug('Imported TensorFlow.')
            # Summary writer and summary path
            summary_path = os.path.join(args.log_dir, '%i' % time.time())
            logger.info('Summary are written to %s' % summary_path)
            writer = summary.FileWriter(summary_path, flush_secs=10)
        except ImportError:
            logger.warn('TensorFlow cannot be imported. TensorBoard summaries will not be generated. Consider to install the CPU-version TensorFlow.')
            args.write_summary = False
            writer = None

    # Set up the environment
    env_id = args.environment
    env = env_registry[env_id].make()
    logger.info('Environment id %s', env_id)
    logger.info('Environment observation space %r', env.observation_space)
    logger.info('Environment action space %r', env.action_space)
    logger.info('Environment reward range %r', env.reward_range)

    # Fix random seeds
    if args.seed is not None:
        logger.info('Set random seeds to %i' % args.seed)
        env.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Set up the method, model and optimizer
    method_key = args.model.split('.')[0]
    met_cls = method_registry[method_key]
    mod_cls = model_registry[args.model]
    opt_cls = optimizer_registry[args.optimizer]
    logger.info('Using method %s' % met_cls)
    logger.info('Using model %s' % mod_cls)
    logger.info('Using optimizer %s' % opt_cls)

    # Initialize
    mod = mod_cls(env.observation_space, env.action_space)
    target_mod = mod_cls(env.observation_space, env.action_space)
    copy_params(mod, target_mod)
    # Show model statistics
    param_count = 0
    for name, param in mod.named_parameters():
        param_count += param.nelement()
        logger.info('Parameter %s size %r', name, param.size())
    logger.info('Parameters has %i elements.', param_count)
    opt = opt_cls(params=mod.parameters(), lr=args.learning_rate)
    tra = met_cls(env, mod, target_mod, opt, writer)

    # Training
    tra.train_for(max_ticks=args.max_ticks, batch_size=args.batch_size, episode_report_interval=args.episode_report_interval, step_report_interval=args.step_report_interval)

    # Wrap up
    if args.write_summary:
        writer.close()
