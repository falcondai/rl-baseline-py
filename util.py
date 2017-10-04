import logging
from gym.envs.classic_control import CartPoleEnv

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

# Simulation for CartPole-v1
def get_cartpole_state(wrapped_cartpole_env):
    assert isinstance(wrapped_cartpole_env.unwrapped, CartPoleEnv), 'Only accepts CartPole-v1 environment.'
    # Unwrap a wrapped CartPoleEnv
    return (wrapped_cartpole_env._elapsed_steps, wrapped_cartpole_env.unwrapped.state)

def set_cartpole_state(wrapped_cartpole_sim, state):
    assert isinstance(wrapped_cartpole_sim.unwrapped, CartPoleEnv), 'Only accepts CartPole-v1 environment.'
    elapsed_steps, cartpole_state = state
    wrapped_cartpole_sim.unwrapped.state = cartpole_state
    wrapped_cartpole_sim.unwrapped.steps_beyond_done = None
    wrapped_cartpole_sim._elapsed_steps = elapsed_steps

def q_from_rollout(sim, state, action, rollout_policy):
    # Start simulator at the current state
    ob = sim.reset()
    set_cartpole_state(sim, state)
    # And the current action
    ob, r, done, extra = sim.step(action)
    total_return = r
    while not done:
        a = rollout_policy(ob)
        ob, r, done, extra = sim.step(a)
        total_return += r
    return total_return

def report_per_episode(write_summary, writer, interval, episode, total_length, total_return):
    # Report once per interval
    if episode % interval == 0:
        logger.info('Episode %i length %i return %g', episode, total_length, total_return)
        if write_summary:
            try:
                # Tensorflow imports for writing summaries
                from tensorflow import Summary
                episode_summary_proto = Summary(value=[
                    Summary.Value(tag='episodic/total_length', simple_value=total_length),
                    Summary.Value(tag='episodic/total_return', simple_value=total_return),
                ])
                writer.add_summary(episode_summary_proto, global_step=episode)
            except ImportError:
                pass
    else:
        logger.debug('Episode %i length %i return %g', episode, total_length, total_return)
