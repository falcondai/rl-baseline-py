import inspect
from gym.envs.registration import registry as gym_env_registry
from torch import optim

from rl_baseline.registry import env_registry, optimizer_registry, method_registry, model_registry, sim_registry
from rl_baseline.core import Spec, GymEnvSpecWrapper
from rl_baseline.methods import a2c, dqn
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
    Spec('sim/gym.CartPole-v0', CartPoleSim, spec=env_registry['gym.CartPole-v0']),
    'gym.CartPole-v0')
sim_registry.register_to(
    Spec('sim/gym.CartPole-v1', CartPoleSim, spec=env_registry['gym.CartPole-v1']),
    'gym.CartPole-v1')

# Atari games
for game in ['air_raid', 'alien', 'amidar', 'assault', 'asterix', 'asteroids', 'atlantis',
    'bank_heist', 'battle_zone', 'beam_rider', 'berzerk', 'bowling', 'boxing', 'breakout', 'carnival',
    'centipede', 'chopper_command', 'crazy_climber', 'demon_attack', 'double_dunk',
    'elevator_action', 'enduro', 'fishing_derby', 'freeway', 'frostbite', 'gopher', 'gravitar',
    'hero', 'ice_hockey', 'jamesbond', 'journey_escape', 'kangaroo', 'krull', 'kung_fu_master',
    'montezuma_revenge', 'ms_pacman', 'name_this_game', 'phoenix', 'pitfall', 'pong', 'pooyan',
    'private_eye', 'qbert', 'riverraid', 'road_runner', 'robotank', 'seaquest', 'skiing',
    'solaris', 'space_invaders', 'star_gunner', 'tennis', 'time_pilot', 'tutankham', 'up_n_down',
    'venture', 'video_pinball', 'wizard_of_wor', 'yars_revenge', 'zaxxon']:

    # space_invaders should yield SpaceInvaders-v0 and SpaceInvaders-ram-v0
    name = ''.join([g.capitalize() for g in game.split('_')])
    key = 'gym.%s-v0' % name
    sim_registry.register_to(
        Spec(key, AtariSim, spec=env_registry[key]),
        key,
    )
