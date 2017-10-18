from six.moves import xrange

import logging, shutil
from operator import xor

import numpy as np
import torch
from torch import nn

from rl_baseline.registration import env_registry, model_registry
from rl_baseline.util import log_format, fix_random_seeds, write_tb_event, get_latest_checkpoint, create_tb_writer, report_perf, extract_checkpoint_t
from rl_baseline.common import evaluate_policy


logging.basicConfig(format=log_format)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

if __name__ == '__main__':
    import argparse, os, glob

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--log-dir', help='Path to the log directory containing checkpoints.')
    parser.add_argument('-c', '--checkpoint', help='Path to a specific checkpoint.')
    parser.add_argument('-n', '--n-episodes', type=int, default=10, help='Number of episodes to sample.')
    parser.add_argument('-e', '--environment', default='gym.CartPole-v0', help='Environment id.')
    parser.add_argument('-m', '--model', default='dqn.mlp', help='Model name.')
    parser.add_argument('-i', '--ignore', dest='ignore_saved_args', action='store_true', help='Ignore the model arguments in the checkpoint.')
    parser.add_argument('--render', action='store_true', help='Show the environment.')
    parser.add_argument('-s', '--seed', type=int, help='Random seed.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show more logs.')

    # Watch mode
    parser.add_argument('-w', '--watch', action='store_true', help='Watch a log directory to evaluate latest checkpoints.')
    parser.add_argument('--no-best', dest='save_best_model', action='store_false', help='Do not save the best model.')
    parser.add_argument('--no-summary', dest='write_summary', action='store_false', help='Do not write summary.')

    args, extra_args = parser.parse_known_args()

    # Verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    logger.debug('Parsed args %r', args)

    # Check for conflicting arguments
    # Evaluating a non-trainable policy
    if not issubclass(model_registry[args.model], nn.Module):
        assert args.checkpoint is None and args.log_dir is None, 'No checkpoint is needed for non-trainable policies.'
        assert args.watch is False, 'Cannot use watch mode with non-trainable policies.'
        use_checkpoint = False
    else:
        assert bool(args.checkpoint) ^ bool(args.log_dir), 'Please only provide path to the log directory OR a specific checkpoint.'
        assert args.log_dir if args.watch else True, 'Watch mode requires `--log-dir` argument to present.'
        use_checkpoint = True

    # Init
    env = env_registry[args.environment].make()
    mod_cls = model_registry[args.model]
    # Use model arguments specified via CLI
    mod_parser = mod_cls.build_parser(prefix='m')
    mod_args, extra_args = mod_parser.parse_known_args(extra_args)
    logger.debug('Parsed model args %r', mod_args)

    if not use_checkpoint or args.ignore_saved_args:
        mod_args_dict = vars(mod_args)

    if len(extra_args) > 0:
        logger.warn('Ignoring extra arguments %r', extra_args)

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
        last_t = 0
        while True:
            # Find the first unevaluated checkpoint
            paths = glob.glob(os.path.join(args.log_dir, '*.pt'))
            paths = sorted(paths, key=extract_checkpoint_t)
            unchecked_checkpoint_paths = [path for path in paths if last_t < extract_checkpoint_t(path)]
            # Wait for the first checkpoint file to show up
            # This is needed if we run eval before training starts
            if len(unchecked_checkpoint_paths) > 0:
                # The first unevaluated checkpoint
                last_checkpoint_path = unchecked_checkpoint_paths[0]
                checkpoint_t = extract_checkpoint_t(last_checkpoint_path)
                try:
                    # Note that checkpoint might be incomplete
                    checkpoint = torch.load(last_checkpoint_path)
                    logger.info('Loading model from %s', last_checkpoint_path)
                    # Use the saved model arguments
                    if use_checkpoint and not args.ignore_saved_args:
                        mod_args_dict = checkpoint['model_args'] or vars(mod_args)
                    mod = mod_cls(env.observation_space, env.action_space, **mod_args_dict)
                    mod.load_state_dict(checkpoint['model'])
                    # Evaluate the checkpoint
                    rets, lens = evaluate_policy(env, mod, args.n_episodes, args.render)
                    report_perf(rets, lens)
                    avg_ret = np.mean(rets)
                    # Keep the best model
                    if args.save_best_model:
                        if best_return is None or best_return < avg_ret:
                            logger.info('New best model with return %g', avg_ret)
                            # Add link to the latest best model
                            os.symlink(last_checkpoint_path, os.path.join(best_model_dir, os.path.basename(last_checkpoint_path)))
                            best_return = avg_ret
                    # Write eval summaries for TensorBoard
                    if args.write_summary:
                        write_tb_event(writer, checkpoint['tick'], {
                            'metrics/episode_return': avg_ret,
                        })
                    last_t = checkpoint_t
                except Exception as e:
                    logger.warn(e)
    else:
        # Single-run evaluation
        if issubclass(model_registry[args.model], nn.Module):
            # Load checkpoint
            if args.log_dir is not None:
                # Load the latest checkpoint from the log directory
                checkpoint_path = get_latest_checkpoint(args.log_dir)
                assert checkpoint_path != None, 'There is no *.pt checkpoint files at %s' % args.log_dir
            else:
                checkpoint_path = args.checkpoint
            logger.info('Loading model from %s', checkpoint_path)
            checkpoint = torch.load(checkpoint_path)
            # Use the saved model arguments
            if use_checkpoint and not args.ignore_saved_args:
                mod_args_dict = checkpoint['model_args'] or vars(mod_args)
            mod = mod_cls(env.observation_space, env.action_space, **mod_args_dict)
            checkpoint['model']['fc1.weight'] = checkpoint['model']['fc1.weight'][:2]
            checkpoint['model']['fc1.bias'] = checkpoint['model']['fc1.bias'][:2]
            mod.load_state_dict(checkpoint['model'])
        else:
            # Non-trainable models
            mod = mod_cls(env.observation_space, env.action_space)

        # Fix random seed
        if args.seed is not None:
            fix_random_seeds(args.seed, env, torch, np)

        # Evaluate
        rets, lens = evaluate_policy(env, mod, args.n_episodes, args.render)
        report_perf(rets, lens)
