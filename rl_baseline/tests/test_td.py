from functools import partial
import numpy as np
import torch
from torch.optim import SGD
from rl_baseline.methods.td import QTrainer, SarsaTrainer, DqnTab
from rl_baseline.methods.dqn import DqnMlp
from rl_baseline.common import evaluate_policy
from rl_baseline.util import polynomial_schedule

def test_td_q():
    from gym.envs.toy_text import FrozenLakeEnv
    from gym.wrappers import TimeLimit
    import gym
    env = TimeLimit(FrozenLakeEnv(is_slippery=False, map_name='4x4'), max_episode_steps=100)
    # env = TimeLimit(FrozenLakeEnv(is_slippery=True, map_name='8x8'), max_episode_steps=100)
    model = DqnTab(env.observation_space, env.action_space, initial_q=1)
    # env = gym.make('CartPole-v0')
    # model = DqnMlp(env.observation_space, env.action_space, hiddens=[16], activation_fn='elu')
    opt = SGD(params=model.parameters(), lr=0.1)
    tra = QTrainer(
        env=env,
        model=model,
        optimizer=opt,
        td_lambda=0.6,
        gamma=0.99,
        eligibility_type='accumulating',
        exploration_type='softmax',
        initial_exploration=0.1,
        terminal_exploration=0.1,
        exploration_start=0,
        exploration_length=1000,
        step_size_schedule=partial(polynomial_schedule, 0.5, 1, 0.00001, 0, 1000))
    np.random.seed(123)
    torch.manual_seed(123)
    tra.train_for(1000)
    rets, lens = evaluate_policy(env, model, 1, False)
    assert rets[0] == 1, 'TD Q should solve this simple maze.'

def test_td_sarsa():
    from gym.envs.toy_text import FrozenLakeEnv
    from gym.wrappers import TimeLimit
    import gym
    # env = TimeLimit(FrozenLakeEnv(is_slippery=False, map_name='4x4'), max_episode_steps=100)
    env = TimeLimit(FrozenLakeEnv(is_slippery=False, map_name='8x8'), max_episode_steps=100)
    model = DqnTab(env.observation_space, env.action_space, initial_q=1)
    # env = gym.make('CartPole-v0')
    # model = DqnMlp(env.observation_space, env.action_space, hiddens=[16], activation_fn='elu')
    opt = SGD(params=model.parameters(), lr=1)
    tra = SarsaTrainer(
        env=env,
        model=model,
        optimizer=opt,
        td_lambda=0.6,
        gamma=0.99,
        eligibility_type='accumulating',
        exploration_type='epsilon',
        initial_exploration=0.1,
        terminal_exploration=0.1,
        exploration_start=0,
        exploration_length=1000,
        step_size_schedule=partial(polynomial_schedule, 0.5, 1, 0.00001, 0, 1000))
    np.random.seed(123)
    torch.manual_seed(123)
    tra.train_for(10000)
    rets, lens = evaluate_policy(env, model, 100, False)
    print(model.q_values.weight)
    assert rets[0] == 1, 'TD Sarsa should solve this simple maze.'

test_td_sarsa()
