import logging, os, glob

import torch

# Set up logging format
log_format = '[%(asctime)s %(filename)s:%(lineno)d] %(levelname)s: %(message)s'
logging.basicConfig(format=log_format)
logger = logging.getLogger(__name__)
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

def make_checkpoint(tick, episode, step, optimizer, model, extra={}):
    '''Construct a dictionary that contains the complete training state.'''
    return {
        'tick': tick,
        'episode': episode,
        'step': step,
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        'extra': extra,
    }

def save_checkpoint(tick, episode, step, optimizer, model, log_dir, extra={}):
    checkpoint = make_checkpoint(
        tick=tick,
        episode=episode,
        step=step,
        optimizer=optimizer,
        model=model,
        extra=extra,
    )
    checkpoint_path = os.path.join(log_dir, 'checkpoint-%i.pt' % tick)
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
        logger.info('Summary are written to %s' % summary_dir)
        writer = summary.FileWriter(summary_dir, flush_secs=10)
    except ImportError:
        logger.warn('TensorFlow cannot be imported. TensorBoard summaries will not be generated. Consider to install the CPU-version TensorFlow.')
        writer = None
    return writer
