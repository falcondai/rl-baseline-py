def test_a2c_on_cartpole():
    # Train a simple A2C linear model to solve CartPole-v0
    from methods.a2c import A2CTrainer, A2CLinearModel
    from util import report_per_episode
    from functools import partial
    import gym
    from torch import optim
    env = gym.make('CartPole-v0')
    mod = A2CLinearModel(env.observation_space, env.action_space)
    opt = optim.SGD(params=mod.parameters(), lr=0.01)
    tra = A2CTrainer(env, mod, opt, partial(report_per_episode, False, None, 1))

    # Train for a little
    tra.train_for(1000, 20, False)
    
