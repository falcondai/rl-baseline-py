def test_dqn_model_parser():
    from rl_baseline.methods.dqn import DqnModel

    parser = DqnModel.build_parser(prefix='m')
    args = parser.parse_args(['--m-exploration', 'epsilon'])
    assert args.exploration_type == 'epsilon', '`DqnModel` parser should handle `exploration_type`.'

def test_dqn_on_cartpole():
    # Train a simple DQN model to solve CartPole-v0
    import gym
    import torch
    import numpy as np
    from torch import optim

    from rl_baseline.methods.dqn import DqnTrainer, DqnMlp
    from rl_baseline.common import evaluate_policy
    from rl_baseline.util import copy_params, fix_random_seeds

    # Fix seed for replication
    seed = 123

    env = gym.make('CartPole-v0')
    fix_random_seeds(seed, env, torch, np)

    mod = DqnMlp(env.observation_space, env.action_space, hiddens=[64], activation_fn='elu')
    target_mod = DqnMlp(env.observation_space, env.action_space, hiddens=[64], activation_fn='elu')
    copy_params(mod, target_mod)

    opt = optim.SGD(params=mod.parameters(), lr=0.1)
    tra = DqnTrainer(
        env=env,
        model=mod,
        target_model=target_mod,
        optimizer=opt,
        capacity=10000,
        criterion='huber',
        max_grad_norm=10,
        exploration_type='epsilon',
        minimal_replay_buffer_occupancy=4000,
        initial_exploration=1,
        terminal_exploration=0.01,
        exploration_length=10000,
        target_update_interval=200,
        double_dqn=False)

    # Train for a little
    tra.train_for(10000, 1, step_report_interval=100, episode_report_interval=100)

    # Evaluate the trained policy
    rets, lens = evaluate_policy(env, mod, n_episodes=100, render=False)
    # Should be `176.31`
    # print(np.mean(rets))
    assert np.mean(rets) > 150, 'Under this simple setting DQN should out perform random policy.'

def test_dqn_on_frozen_lake():
    # Train a simple DQN model to solve CartPole-v0
    import gym
    import torch
    import numpy as np
    from torch import optim

    from rl_baseline.methods.dqn import DqnTrainer, DqnTab
    from rl_baseline.common import evaluate_policy
    from rl_baseline.util import copy_params, fix_random_seeds

    # Fix seed for replication
    seed = 123

    env = gym.make('FrozenLake-v0')
    fix_random_seeds(seed, env, torch, np)

    mod = DqnTab(env.observation_space, env.action_space, 1)
    target_mod = DqnTab(env.observation_space, env.action_space, None)
    copy_params(mod, target_mod)

    opt = optim.SGD(params=mod.parameters(), lr=0.1)
    tra = DqnTrainer(
        env=env,
        model=mod,
        target_model=target_mod,
        optimizer=opt,
        capacity=10000,
        criterion='huber',
        max_grad_norm=10,
        exploration_type='epsilon',
        minimal_replay_buffer_occupancy=4000,
        initial_exploration=1,
        terminal_exploration=0.01,
        exploration_length=10000,
        target_update_interval=200,
        double_dqn=True)

    # Train for a little
    tra.train_for(20000, update_interval=1, step_report_interval=100, episode_report_interval=100)

    # Evaluate the trained policy
    rets, lens = evaluate_policy(env, mod, n_episodes=100, render=False)
    # print(np.mean(rets))
    # Should be `0.45`
    assert 0.4 < np.mean(rets), 'Under this simple setting DQN should out perform random policy.'
