import logging
from gym.envs.classic_control import CartPoleEnv

# Set up logging format
log_format = '[%(asctime)s %(filename)s:%(lineno)d] %(levelname)s: %(message)s'
logging.basicConfig(format=log_format)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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
