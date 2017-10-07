from rl_baseline.core import GymEnvSpecWrapper

def test_gym_env_spec_wrapper():
    from gym.envs.registration import registry
    from gym.wrappers.time_limit import TimeLimit

    spec = GymEnvSpecWrapper(registry.spec('CartPole-v1'))
    env = spec.make()

    assert isinstance(env, TimeLimit), 'The gym environment should be wrapped by `TimeLimit` wrapper.'
