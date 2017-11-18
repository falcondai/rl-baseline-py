import numpy as np
from rl_baseline.methods.td import QTrainer, SarsaTrainer

def test_td_q():
    from gym.envs.toy_text import FrozenLakeEnv
    env = FrozenLakeEnv(is_slippery=False, map_name='4x4')
    model = np.ones((env.observation_space.n, env.action_space.n))
    tra = QTrainer(env=env, model=model, td_lambda=0.6, gamma=0.99, eligibility_type='accumulating', exploration_type='epsilon', initial_exploration=0.1, terminal_exploration=0.1, exploration_start=0, exploration_length=1000, initial_step_size=1, step_size_schedule=lambda tick : 1 / tick ** 0.5)
    tra.train_for(5000)

def test_td_sarsa():
    from gym.envs.toy_text import FrozenLakeEnv
    env = FrozenLakeEnv(is_slippery=False, map_name='4x4')
    model = np.ones((env.observation_space.n, env.action_space.n))
    tra = SarsaTrainer(env=env, model=model, td_lambda=0.6, gamma=0.99, eligibility_type='accumulating', exploration_type='epsilon', initial_exploration=0.1, terminal_exploration=0.1, exploration_start=0, exploration_length=1000, initial_step_size=1, step_size_schedule=lambda tick : 1 / tick ** 0.5)
    tra.train_for(5000)
