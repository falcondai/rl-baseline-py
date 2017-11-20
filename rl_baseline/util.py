import logging, os, glob

import numpy as np
import torch

# Set up logging format
log_format = '%(levelname)s [%(asctime)s %(filename)s:%(lineno)d] %(message)s'
logging.basicConfig(format=log_format)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

def global_norm(parameters):
    squares = 0
    for parameter in parameters:
        squares += parameter.norm() ** 2
    return squares.sqrt()

def write_tb_event(writer, tick, kv_pairs):
    try:
        # Tensorflow imports for writing summaries
        from tensorflow import Summary
        episode_summary_proto = Summary(value=[Summary.Value(tag=k, simple_value=v) for k, v in kv_pairs.items()])
        writer.add_summary(episode_summary_proto, global_step=tick)
    except ImportError:
        pass

def linear_schedule(start_y, end_y, start_t, end_t, t):
    if t < start_t:
        return start_y
    if end_t < t:
        return end_y
    return (end_y - start_y) * (t - start_t) / (end_t - start_t) + start_y

def polynomial_schedule(p, start_y, end_y, start_t, end_t, t):
    '''Return t ^ -p'''
    if t < start_t:
        return start_y
    if end_t < t:
        return end_y
    t = min(end_t, t)
    return end_y + (start_y - end_y) * (1 - t / end_t) ** p

def copy_params(from_model, to_model):
    to_model.load_state_dict(from_model.state_dict())

class Saver:
    '''Keeps track of some important objects and makes checkpointing more convenient.'''
    def __init__(self, log_dir, model, optimizer, model_id, model_args, method_args):
        '''
        Args:
            model_id : str
                Name of the model in the model registry.
            model_args : dict
                Arguments to pass to a model's constructor.
            method_args : dict
                Arguments to pass to a trainer's constructor.
        '''
        self.log_dir = log_dir
        self.model = model
        self.model_id = model_id
        self.optimizer = optimizer
        self.model_args = model_args
        self.method_args = method_args

        # Create the log directory
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            logger.debug('Created the log directory %s', self.log_dir)

    def save_checkpoint(self, tick, episode, step):
        checkpoint = {
            'tick': tick,
            'episode': episode,
            'step': step,
            'optimizer': self.optimizer.state_dict(),
            'model': self.model.state_dict(),
            'model_id': self.model_id,
            'model_args': self.model_args,
            'method_args': self.method_args,
        }
        checkpoint_path = os.path.join(self.log_dir, 'checkpoint-%i.pt' % tick)
        torch.save(checkpoint, checkpoint_path)
        logger.info('Saved checkpoint at %s', checkpoint_path)

def fix_random_seeds(seed, env, torch, numpy):
    logger.info('Set random seeds to %i' % seed)
    env.seed(seed)
    torch.manual_seed(seed)
    numpy.random.seed(seed)

def extract_checkpoint_t(checkpoint_path):
    # Assume checkpoints are saved to paths like /tmp/dqn/checkpoint-11501.pt
    # XXX the modified time via getmtime sometimes causes eval to skip the last checkpoint
    return int(checkpoint_path.split('-')[-1][:-3])

def get_latest_checkpoint(log_dir):
    pt_paths = glob.glob(os.path.join(log_dir, '*.pt'))
    if len(pt_paths) == 0:
        logger.warn('There is no *.pt checkpoint files at %s' % log_dir)
        return None
    if 1 < len(pt_paths):
        latest_checkpoint_path = max(pt_paths, key=extract_checkpoint_t)
        return latest_checkpoint_path
    # There is only one checkpoint (best model folder)
    return pt_paths[0]

def create_tb_writer(summary_dir):
    try:
        # Tensorflow imports for writing summaries
        from tensorflow import summary
        logger.debug('Imported TensorFlow.')
        # Summary writer and summary path
        logger.info('TensorBoard summary is being written to %s' % summary_dir)
        writer = summary.FileWriter(summary_dir, flush_secs=10)
    except ImportError:
        logger.warn('TensorFlow cannot be imported. TensorBoard summaries will not be generated. Consider to install the CPU-version TensorFlow.')
        writer = None
    return writer

def report_perf(returns, lengths, log_level=logging.INFO):
    logger.log(log_level, 'Total %i episodes', len(returns))
    logger.log(log_level, 'Episode return mean/max/min/median %g/%g/%g/%g', np.mean(returns), np.max(returns), np.min(returns), np.median(returns))
    logger.log(log_level, 'Episode length mean/max/min/median %g/%g/%g/%g', np.mean(lengths), np.max(lengths), np.min(lengths), np.median(lengths))

def report_model_stats(model):
    logger.info('Model architecture %r', model)
    param_count = 0
    for name, param in model.named_parameters():
        param_count += param.nelement()
        logger.info('Parameter %s size %r', name, param.size())
    logger.info('Parameters has %i elements.', param_count)
