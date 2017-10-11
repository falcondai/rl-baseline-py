import logging, os, glob

import numpy as np
import torch

# Set up logging format
log_format = '[%(asctime)s %(filename)s:%(lineno)d] %(levelname)s: %(message)s'
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

def copy_params(from_model, to_model):
    to_model.load_state_dict(from_model.state_dict())

class Saver:
    '''Keeps track of some important objects and makes checkpointing more convenient.'''
    def __init__(self, log_dir, model, optimizer, model_args, method_args):
        self.log_dir = log_dir
        self.model = model
        self.optimizer = optimizer
        self.model_args = model_args
        self.method_args = method_args

    def save_checkpoint(self, tick, episode, step):
        checkpoint = {
            'tick': tick,
            'episode': episode,
            'step': step,
            'optimizer': self.optimizer.state_dict(),
            'model': self.model.state_dict(),
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

def get_latest_checkpoint(log_dir):
    pt_paths = glob.glob(os.path.join(log_dir, '*.pt'))
    if len(pt_paths) == 0:
        logger.warn('There is no *.pt checkpoint files at %s' % log_dir)
        return None
    latest_checkpoint_path = max(pt_paths, key=os.path.getmtime)
    return latest_checkpoint_path

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
