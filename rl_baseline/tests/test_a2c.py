def test_a2c_on_cartpole():
    # Train a simple A2C linear model to solve CartPole-v0
    import gym
    import torch
    import numpy as np
    from torch import optim

    from rl_baseline.methods.a2c import A2cTrainer, A2cLinearModel
    from rl_baseline.common import evaluate_policy

    # Fix seed for replication
    seed = 777
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make('CartPole-v0')
    env.seed(seed)

    mod = A2cLinearModel(env.observation_space, env.action_space)
    opt = optim.SGD(params=mod.parameters(), lr=0.01)
    tra = A2cTrainer(env, mod, opt)

    # Train for a little
    tra.train_for(500, 20, step_report_interval=10)

    # Evaluate the trained policy
    rets, lens = evaluate_policy(env, mod, n_episodes=100, render=False)


test_a2c_on_cartpole()
