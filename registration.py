import inspect
from core import GymEnvSpecWrapper
from registry import env_registry, optimizer_registry, method_registry, model_registry
from gym.envs.registration import registry
from torch import optim
from methods import a2c, dqn

# Register all envs in Gym
for spec in registry.all():
    spec = GymEnvSpecWrapper(spec)
    env_registry.register_to(spec, 'gym.%s' % spec.id)

# Register all optimizers in torch.optim
for kls_name, kls in inspect.getmembers(optim):
    # Filter out all optimizer classes
    if inspect.isclass(kls):
        optimizer_registry.register_to(kls, kls_name)
