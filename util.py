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

def write_tb_event(writer, t, kv_pairs):
    try:
        # Tensorflow imports for writing summaries
        from tensorflow import Summary
        episode_summary_proto = Summary(value=[Summary.Value(tag=k, simple_value=v) for k, v in kv_pairs.items()])
        writer.add_summary(episode_summary_proto, global_step=t)
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
