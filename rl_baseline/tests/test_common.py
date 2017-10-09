def test_exhaustive_policy_cartpole():
    from rl_baseline.common import RandomPolicy, ExhaustivePolicy, evaluate_policy
    from rl_baseline.sims.cartpole import CartPoleSim
    import numpy as np
    import gym
    gym.undo_logger_setup()

    env = gym.make('CartPole-v0')
    sim = CartPoleSim(env.spec)
    np.random.seed(123)
    env.seed(123)

    policy = ExhaustivePolicy(env, sim, rollout_policy=RandomPolicy(env.observation_space, env.action_space), max_rollouts=200, search_bound=20)

    rets, lens = evaluate_policy(env, policy, 1, render=False)
    assert rets[0] == 200, 'Exhaustive policy should reach perfect performance.'
