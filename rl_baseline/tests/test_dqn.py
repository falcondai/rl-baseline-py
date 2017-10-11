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
        target_update_interval=200)

    # Train for a little
    tra.train_for(20000, 20, step_report_interval=100, episode_report_interval=100)

    # Evaluate the trained policy
    rets, lens = evaluate_policy(env, mod, n_episodes=100, render=False)
    assert np.mean(rets) > 45, 'Under this simple setting DQN should out performance random policy.'

test_dqn_on_cartpole()
