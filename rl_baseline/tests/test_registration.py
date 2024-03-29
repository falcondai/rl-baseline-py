from rl_baseline.core import GymEnvSpecWrapper
from rl_baseline.registration import env_registry

def test_gym_specs_wrapped():
    spec = env_registry['gym.CartPole-v1']
    assert isinstance(spec, GymEnvSpecWrapper), 'Gym environments should have specs wrapped by GymEnvSpecWrapper.'
