def test_a2c_on_cartpole():
    # Train a simple A2C linear model to solve CartPole-v0
    import gym
    import torch
    import numpy as np
    from torch import optim

    from rl_baseline.methods.a2c import A2CTrainer, A2CLinearModel

    # Fix seed for replication
    seed = 777
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make('CartPole-v0')
    env.seed(seed)

    mod = A2CLinearModel(env.observation_space, env.action_space)
    opt = optim.SGD(params=mod.parameters(), lr=0.01)
    tra = A2CTrainer(env, mod, opt)

    # Train for a little
    tra.train_for(1000, 20)

    # TODO evaluate and assert its performance
