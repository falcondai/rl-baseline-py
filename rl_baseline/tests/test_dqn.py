def test_dqn_on_cartpole():
    # Train a simple DQN model to solve CartPole-v0
    import gym
    import torch
    import numpy as np
    from torch import optim

    from rl_baseline.methods.dqn import DqnTrainer, DqnMlp

    # Fix seed for replication
    seed = 777
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make('CartPole-v0')
    env.seed(seed)

    mod = DqnMlp(env.observation_space, env.action_space)
    target_mod = DqnMlp(env.observation_space, env.action_space)

    opt = optim.SGD(params=mod.parameters(), lr=0.01)
    tra = DqnTrainer(env, mod, target_model=target_mod, optimizer=opt)

    # Train for a little
    tra.train_for(100, 20, step_report_interval=20, minimal_replay_buffer_occupancy=10)

    # TODO evaluate and assert its performance
