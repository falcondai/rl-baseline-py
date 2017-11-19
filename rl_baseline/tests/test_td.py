import numpy as np
from rl_baseline.methods.td import QTrainer, SarsaTrainer, DqnTab
from torch.optim import SGD
from rl_baseline.common import evaluate_policy

def test_td_q():
    from gym.envs.toy_text import FrozenLakeEnv
    env = FrozenLakeEnv(is_slippery=False, map_name='4x4')
    model = np.ones((env.observation_space.n, env.action_space.n))
    tra = QTrainer(env=env, model=model, td_lambda=0.6, gamma=0.99, eligibility_type='accumulating', exploration_type='epsilon', initial_exploration=0.1, terminal_exploration=0.1, exploration_start=0, exploration_length=1000, initial_step_size=1, step_size_schedule=lambda tick : 1 / tick ** 0.5)
    tra.train_for(5000)

def test_td_sarsa():
    from gym.envs.toy_text import FrozenLakeEnv
    from gym.wrappers import TimeLimit
    env = TimeLimit(FrozenLakeEnv(is_slippery=False, map_name='4x4'), max_episode_steps=100)
    model = DqnTab(env.observation_space, env.action_space, initial_q=1)
    opt = SGD(params=model.parameters(), lr=1)
    tra = SarsaTrainer(env=env, model=model, optimizer=opt, td_lambda=0.6, gamma=0.99, eligibility_type='replacing', exploration_type='epsilon', initial_exploration=0.1, terminal_exploration=0.1, exploration_start=0, exploration_length=1000, step_size_schedule=lambda tick : 1 / (tick + 1) ** 0.5)
    np.random.seed(123)
    tra.train_for(1000)
    rets, lens = evaluate_policy(env, model, 1, False)
    assert rets[0] == 1, 'TD Sarsa should solve this simple maze.'

test_td_sarsa()
