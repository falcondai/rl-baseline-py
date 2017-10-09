from six.moves import xrange

import logging, shutil
from operator import xor

import numpy as np
import torch
from torch import nn

from rl_baseline.registration import env_registry, model_registry
from rl_baseline.util import log_format, fix_random_seeds, write_tb_event, get_latest_checkpoint, create_tb_writer
from rl_baseline.common import evaluate_policy


logging.basicConfig(format=log_format)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__ == '__main__':
    import argparse, os, glob

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--log-dir', help='Path to the log directory containing checkpoints.')
    parser.add_argument('-c', '--checkpoint', help='Path to a specific checkpoint.')
    parser.add_argument('-n', '--n-episodes', type=int, default=10, help='Number of episodes to sample.')
    parser.add_argument('-e', '--environment', default='gym.CartPole-v0', help='Environment id.')
    parser.add_argument('-m', '--model', default='dqn.mlp', help='Model name.')
    parser.add_argument('--render', action='store_true', help='Show the environment.')
    parser.add_argument('-s', '--seed', type=int, help='Random seed.')

    # Watch mode
    parser.add_argument('-w', '--watch', action='store_true', help='Watch a log directory to evaluate latest checkpoints.')
    parser.add_argument('--no-best', dest='save_best_model', action='store_false', help='Do not save the best model.')
    parser.add_argument('--no-summary', dest='write_summary', action='store_false', help='Do not write summary.')

    args = parser.parse_args()

    # Check for conflicting arguments
    # Evaluating a non-trainable policy
    if not issubclass(model_registry[args.model], nn.Module):
        assert args.checkpoint is None and args.log_dir is None, 'No checkpoint is needed for non-trainable policies.'
        assert args.watch is False, 'Cannot use watch mode with non-trainable policies.'
    else:
        assert bool(args.checkpoint) ^ bool(args.log_dir), 'Please only provide path to the log directory OR a specific checkpoint.'
        assert args.log_dir if args.watch else True, 'Watch mode requires `--log-dir` argument to present.'

    # Init
    env = env_registry[args.environment].make()
    mod = model_registry[args.model](env.observation_space, env.action_space)

    if args.watch:
        # Watch mode
        # Setup for tracking the best model evaluated so far
        if args.save_best_model:
            best_model_dir = os.path.join(args.log_dir, 'best')
            best_model_path = os.path.join(best_model_dir, 'checkpoint.pt')
            if not os.path.exists(best_model_dir):
                os.makedirs(best_model_dir)
                logger.debug('Created the best model directory %s', best_model_dir)
            best_return = None
        # Setup for writing eval summary
        if args.write_summary:
            eval_summary_dir = os.path.join(args.log_dir, 'eval')
            if not os.path.exists(eval_summary_dir):
                os.makedirs(eval_summary_dir)
                logger.debug('Created the eval summary directory %s', eval_summary_dir)
            writer = create_tb_writer(eval_summary_dir)
        # Start the monitoring loop
        last_mtime = 0
        while True:
            last_checkpoint_path = get_latest_checkpoint(args.log_dir)
            # Wait for the first checkpoint file to show up
            # This is needed if we run eval before training starts
            if last_checkpoint_path is not None:
                mtime = os.path.getmtime(last_checkpoint_path)
                if last_mtime < mtime:
                    try:
                        # Note that checkpoint might be incomplete
                        checkpoint = torch.load(last_checkpoint_path)
                        logger.info('Loading model from %s', last_checkpoint_path)
                        mod.load_state_dict(checkpoint['model'])
                        rets, lens = evaluate_policy(env, mod, args.n_episodes, args.render)
                        avg_ret = np.mean(rets)
                        # Keep the best model
                        if args.save_best_model:
                            if best_return is None or best_return < avg_ret:
                                logger.info('New best model with return %g', avg_ret)
                                # Replace the best model
                                shutil.copyfile(last_checkpoint_path, best_model_path)
                                best_return = avg_ret
                        # Write eval summaries for TensorBoard
                        if args.write_summary:
                            write_tb_event(writer, checkpoint['tick'], {
                                'metrics/episode_return': avg_ret,
                            })
                        last_mtime = mtime
                    except Exception as e:
                        logger.warn(e)
    else:
        # Single-run evaluation
        if isinstance(model_registry[args.model], nn.Module):
            # Load checkpoint
            if args.log_dir is not None:
                # Load the latest checkpoint from the log directory
                checkpoint_path = get_latest_checkpoint(args.log_dir)
                assert checkpoint_path != None, 'There is no *.pt checkpoint files at %s' % args.log_dir
            else:
                checkpoint_path = args.checkpoint
            logger.info('Loading model from %s', checkpoint_path)
            checkpoint = torch.load(checkpoint_path)
            mod.load_state_dict(checkpoint['model'])

        # Fix random seed
        if args.seed is not None:
            fix_random_seeds(args.seed, env, torch, np)

        # Evaluate
        evaluate_policy(env, mod, args.n_episodes, args.render)
