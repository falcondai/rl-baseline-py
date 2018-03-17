import inspect
from gym.envs.registration import registry as gym_env_registry
from torch import optim

from rl_baseline.registry import env_registry, optimizer_registry, method_registry, model_registry, sim_registry
from rl_baseline.core import Spec, GymEnvSpecWrapper
from rl_baseline.methods import a2c, dqn, td
from rl_baseline.envs.atari import AtariDqnEnvWrapper
from rl_baseline.sims.cartpole import CartPoleSim
from rl_baseline.sims.atari import AtariSim


# Register all envs in Gym
for spec in gym_env_registry.all():
    spec = GymEnvSpecWrapper(spec)
    env_registry.register_to(spec, 'gym.%s' % spec.id)

# Register all optimizers in torch.optim
for kls_name, kls in inspect.getmembers(optim):
    # Filter out all optimizer classes
    if inspect.isclass(kls):
        optimizer_registry.register_to(kls, kls_name)

# Register simulators
# CartPole envs
sim_registry.register_to(
    Spec('gym.CartPole-v0', CartPoleSim, spec=env_registry['gym.CartPole-v0']),
    'gym.CartPole-v0')
sim_registry.register_to(
    Spec('gym.CartPole-v1', CartPoleSim, spec=env_registry['gym.CartPole-v1']),
    'gym.CartPole-v1')

# Atari games
for spec in gym_env_registry.all():
    # HACK gym uses class path string as entry point
    if spec._entry_point.endswith(':AtariEnv'):
        # Atari envs in gym
        key = 'gym.%s' % spec.id
        sim_registry.register_to(
            Spec(key, AtariSim, spec=spec),
            key,
        )
        # Register DQN pre-processing as a wrapped environment
        if spec.id.endswith('Deterministic-v4'):
            name = spec.id[:-len('Deterministic-v4')]
            key = 'dqn.%s' % name
            env_registry.register_to(
                Spec(
                    key,
                    lambda : AtariDqnEnvWrapper(gym_env_registry.make('PongDeterministic-v4'))
                ), key)
