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

def test_exhaustive_policy_mountaincar():
    from rl_baseline.common import RandomPolicy, ExhaustivePolicy, evaluate_policy
    from rl_baseline.sims.cartpole import MountainCarSim
    import numpy as np
    import gym
    gym.undo_logger_setup()

    env = gym.make('MountainCar-v0')
    sim = MountainCarSim(env.spec)
    np.random.seed(123)
    env.seed(123)

    policy = ExhaustivePolicy(env, sim, rollout_policy=RandomPolicy(env.observation_space, env.action_space), max_rollouts=20, search_bound=None)

    rets, lens = evaluate_policy(env, policy, 1, render=False)
    print(rets)
    # assert rets[0] == 200, 'Exhaustive policy should reach perfect performance.'

# test_exhaustive_policy_mountaincar()

def test_exhaustive_policy_atari():
    from rl_baseline.common import RandomPolicy, ExhaustivePolicy, evaluate_policy
    from rl_baseline.sims.atari import AtariSim
    import numpy as np
    import gym
    gym.undo_logger_setup()

    env = gym.make('Pong-v0')
    # env = gym.make('MontezumaRevenge-v0')
    sim = AtariSim(env.spec)
    np.random.seed(123)
    env.seed(123)

    policy = ExhaustivePolicy(env, sim, rollout_policy=RandomPolicy(env.observation_space, env.action_space), max_rollouts=10, search_bound=1000)

    rets, lens = evaluate_policy(env, policy, 1, render=True)
    assert rets[0] > -5, 'Bounded exhaustive policy should reach strong performance.'

def test_exhaustive_policy_cts_lunarlander():
    from rl_baseline.common import RandomPolicy, ExhaustivePolicy, evaluate_policy
    # from rl_baseline.sims.cartpole import CartPoleSim
    import numpy as np
    import gym
    gym.undo_logger_setup()

    env = gym.make('LunarLanderContinuous-v2')
    # sim = CartPoleSim(env.spec)
    np.random.seed(123)
    env.seed(123)

    policy = ExhaustivePolicy(env, sim, rollout_policy=RandomPolicy(env.observation_space, env.action_space), max_rollouts=200, search_bound=20)

    rets, lens = evaluate_policy(env, policy, 1, render=False)
    assert rets[0] == 200, 'Exhaustive policy should reach perfect performance.'

# test_exhaustive_policy_atari()
# test_exhaustive_policy_cartpole()
