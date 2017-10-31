from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import logging, itertools, os
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
from torch.nn import functional as f

import gym
gym.undo_logger_setup()

from rl_baseline.registration import env_registry, optimizer_registry, model_registry, method_registry
from rl_baseline.util import log_format, global_norm, copy_params, Saver, fix_random_seeds, create_tb_writer


logging.basicConfig(format=log_format)
logger = logging.getLogger()

# Set the logging level
logger.setLevel(logging.INFO)

if __name__ == '__main__':
    import time, argparse, os

    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--environment', default='gym.CartPole-v0', choices=env_registry.all().keys(), help='Environment id.')
    parser.add_argument('-m', '--model', default='dqn.mlp', choices=model_registry.all().keys(), help='Model id.')
    parser.add_argument('-l', '--log-dir', default='./logs', help='Path to log directory.')
    parser.add_argument('--no-summary', dest='write_summary', action='store_false', help='Do not write summary protobuf for TensorBoard.')
    parser.add_argument('--episode-report-interval', type=int, default=1, help='Report every N-many episodes.')
    parser.add_argument('--step-report-interval', type=int, default=400, help='Report every N-many steps.')
    parser.add_argument('--save-interval', type=int, default=100, help='Save every N-many steps.')
    parser.add_argument('--eval-interval', type=int, default=0, help='Evaluate every N-many steps. 0 disables the periodic evaluation.')
    parser.add_argument('--n-eval-episodes', type=int, default=10, help='Number of episodes used to evaluate.')
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.05, help='Initial learning rate.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed. Accepts float but will be cast to integer.')
    parser.add_argument('-n', '--max-ticks', type=int, default=10**4, help='Maximum number of ticks to train.')
    parser.add_argument('-b', '--batch-size', type=int, default=32, help='Batch size.')
    parser.add_argument('-o', '--optimizer', default='SGD', choices=optimizer_registry.all().keys(), help='Optimizer to use.')
    parser.add_argument('--render', action='store_true', help='Show the environment.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show more logs.')
    # TODO add LR scheduler
    parser.add_argument('--lr-scheduler', default=None, help='Scheduler for learning rates.')
    # TODO restore from checkpoint
    parser.add_argument('-f', '--restore', help='Path to an existing checkpoint.')
    # TODO add GPU support
    parser.add_argument('--gpu', dest='gpu_id', default=None, help='GPU id to use.')

    args, extra_args = parser.parse_known_args()

    # Set the logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Debug information
    logger.debug('PyTorch version %s', torch.__version__)
    logger.debug('Parsed args %r', args)

    # Parse the method-specific arguments
    method_key = args.model.split('.')[0]
    met_cls = method_registry[method_key]
    met_parser = met_cls.build_parser(prefix='m')
    met_args, extra_args = met_parser.parse_known_args(extra_args)
    logger.debug('Parsed method args %r', met_args)

    # Create the log directory
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
        logger.debug('Created the log directory %s', args.log_dir)

    # Summary for TensorBoard monitoring
    if args.write_summary:
        train_summary_dir = os.path.join(args.log_dir, 'train')
        train_summary_writer = create_tb_writer(train_summary_dir)
        eval_summary_dir = os.path.join(args.log_dir, 'eval')
        eval_summary_writer = create_tb_writer(eval_summary_dir)

    # Set up the environment
    env_id = args.environment
    env = env_registry[env_id].make()
    logger.info('Environment id %s', env_id)
    logger.info('Environment observation space %r', env.observation_space)
    logger.info('Environment action space %r', env.action_space)
    logger.info('Environment reward range %r', env.reward_range)

    # Fix random seeds
    if args.seed is not None:
        fix_random_seeds(args.seed, env, torch, np)

    # Set up the method, model and optimizer
    mod_cls = model_registry[args.model]
    opt_cls = optimizer_registry[args.optimizer]

    # Parse model-specific arguments
    mod_parser = mod_cls.build_parser(prefix='m')
    mod_args, extra_args = mod_parser.parse_known_args(extra_args)
    logger.debug('Parsed model args %r', mod_args)

    if len(extra_args) > 0:
        logger.warn('Ignoring extra arguments %r', extra_args)

    logger.info('Using method %s' % met_cls)
    logger.info('Using model %s' % mod_cls)
    logger.info('Using optimizer %s' % opt_cls)

    # Initialize
    mod = mod_cls(env.observation_space, env.action_space, **vars(mod_args))
    target_mod = mod_cls(env.observation_space, env.action_space, **vars(mod_args))
    copy_params(mod, target_mod)
    # Show model statistics
    logger.info('Model architecture %r', mod)
    param_count = 0
    for name, param in mod.named_parameters():
        param_count += param.nelement()
        logger.info('Parameter %s size %r', name, param.size())
    logger.info('Parameters has %i elements.', param_count)
    opt = opt_cls(params=mod.parameters(), lr=args.learning_rate)
    saver = Saver(args.log_dir, mod, opt, model_args=vars(mod_args), method_args=vars(met_args))
    eval_env = env_registry[env_id].make()
    tra = met_cls(
        env=env,
        model=mod,
        target_model=target_mod,
        optimizer=opt,
        eval_env=eval_env,
        train_summary_writer=train_summary_writer,
        eval_summary_writer=eval_summary_writer,
        saver=saver,
        **vars(met_args),
    )

    # Training
    tick, episode, step = tra.train_for(
        max_ticks=args.max_ticks,
        episode_report_interval=args.episode_report_interval,
        step_report_interval=args.step_report_interval,
        checkpoint_interval=args.save_interval,
        eval_interval=args.eval_interval,
        n_eval_episodes=args.n_eval_episodes,
        render=args.render,
    )

    # Wrap up
    if args.write_summary:
        train_summary_writer.close()
        eval_summary_writer.close()

    # Save one last checkpoint
    saver.save_checkpoint(tick, episode, step)
