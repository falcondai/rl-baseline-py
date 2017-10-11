from six.moves import xrange

import logging, itertools, argparse

import numpy as np
from gym import spaces, undo_logger_setup
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as f
from torch.nn.utils import clip_grad_norm

from rl_baseline.core import Policy, StateValue, ActionValue, Parsable
from rl_baseline.registry import method_registry, model_registry, optimizer_registry
from rl_baseline.util import global_norm, log_format, write_tb_event, linear_schedule, copy_params, save_checkpoint


# Set up logger
undo_logger_setup()
logging.basicConfig(format=log_format)
logger = logging.getLogger()
# Set the logging level
logger.setLevel(logging.DEBUG)


class DqnModel(Policy, ActionValue, StateValue, nn.Module, Parsable):
    def __init__(self, *args, **kwargs):
        super(DqnModel, self).__init__(*args, **kwargs)

    def q(self, ob):
        return self.forward(ob)

    def va(self, ob):
        q = self.q(ob)
        max_q, ac = q.max(1)
        return max_q

    def act(self, ob):
        v_ob = Variable(torch.FloatTensor([ob.astype('float')]))
        q = self.q(v_ob)
        max_q, max_ac = q.max(1)
        return max_ac.data[0]


class ReplayBuffer(Parsable):
    '''A circular buffer for storing (s, a, r, s') tuples.'''
    @classmethod
    def add_args(kls, parser, prefix):
        parser.add_argument(kls.prefix_arg_name('capacity', prefix), dest='capacity', type=int, default=5000, help='Size of the replay buffer.')

    def __init__(self, capacity, ob_space, ac_space):
        # TODO handle more spaces
        # TODO test
        obs_shape = [capacity] + list(ob_space.shape)
        self.obs = np.zeros(obs_shape, dtype='float')
        self.acs = np.zeros(capacity, dtype='int')
        self.rs = np.zeros(capacity, dtype='float')
        self.next_obs = np.zeros(obs_shape, dtype='float')
        self.dones = np.zeros(capacity, dtype='bool')
        self._count = 0
        self.capacity = capacity

    @property
    def next_index(self):
        return self._count % self.capacity

    @property
    def occupancy(self):
        return min(self._count, self.capacity)

    def add(self, ob, ac, r, next_ob, done):
        i = self.next_index
        self.obs[i] = ob
        self.acs[i] = ac
        self.rs[i] = r
        self.next_obs[i] = next_ob
        self.dones[i] = done

        self._count += 1

    def sample_sars(self, size):
        '''Sample with replacement (s, a, r, s') tuples of `size`-length arrays.'''
        idxes = [np.random.randint(self.occupancy) for _ in xrange(size)]
        return self.obs[idxes], self.acs[idxes], self.rs[idxes], self.next_obs[idxes], self.dones[idxes]


@method_registry.register('dqn')
class DqnTrainer(Parsable):
    '''Q-learning'''
    @classmethod
    def add_args(kls, parser, prefix):
        parser.add_argument(
            kls.prefix_arg_name('criterion', prefix),
            dest='criterion',
            choices=['l2', 'huber'],
            default='huber',
            help='Loss functional.')
        parser.add_argument(
            kls.prefix_arg_name('max-grad-norm', prefix),
            dest='max_grad_norm',
            type=float,
            default=10,
            help='Maximum norm of the gradient. -1 for no limit.')
        parser.add_argument(
            kls.prefix_arg_name('target-update-interval', prefix),
            dest='target_update_interval',
            type=int,
            default=500,
            help='How many parameter update steps before each target model update.')
        parser.add_argument(
            kls.prefix_arg_name('exp-type', prefix),
            dest='exploration_type',
            choices=['softmax', 'epsilon'],
            default='epsilon',
            help='Type of exploration strategy. Default to be epsilon.')
        parser.add_argument(
            kls.prefix_arg_name('exp-initial', prefix),
            dest='initial_exploration',
            type=float,
            default=1,
            help='Inital value for the exploration factor.')
        parser.add_argument(
            kls.prefix_arg_name('exp-terminal', prefix),
            dest='terminal_exploration',
            type=float,
            default=0.01,
            help='Terminal value for the exploration factor.')
        parser.add_argument(
            kls.prefix_arg_name('exp-length', prefix),
            dest='exploration_length',
            type=int,
            default=1000,
            help='Length of the linear decay schedule of the exloration factor.')
        parser.add_argument(
            kls.prefix_arg_name('min-replay-size', prefix),
            dest='minimal_replay_buffer_occupancy',
            type=int,
            default=100,
            help='Length of the linear decay schedule of the exloration factor.')
        # Add replay buffer's arguments
        ReplayBuffer.add_args(parser, prefix)

    def __init__(self, env, model, target_model, optimizer, capacity, criterion, max_grad_norm, target_update_interval, exploration_type, initial_exploration, terminal_exploration, exploration_length, minimal_replay_buffer_occupancy, writer=None, log_dir=None):
        assert isinstance(model, DqnModel), 'The model argument needs to be an instance of `DqnModel`.'
        assert criterion in ['l2', 'huber'], '`criterion` has to be one of {`l2`, `huber`}.'
        assert exploration_type in ['softmax', 'epsilon'], 'Only supports `softmax` and `epsilon`-greedy exploration strategies.'

        self.env = env
        self.model = model
        self.target_model = target_model
        self.optimizer = optimizer

        # Optional utilities for logging
        self.writer = writer
        self.log_dir = log_dir

        # Total tick count
        self.total_ticks = 0
        self.q_crit = nn.SmoothL1Loss() if criterion == 'huber' else nn.MSELoss()
        self.max_grad_norm = max_grad_norm
        self.exploration_type = exploration_type
        self.initial_exploration = initial_exploration
        self.terminal_exploration = terminal_exploration
        self.exploration_length = exploration_length
        self.target_update_interval = target_update_interval
        self.minimal_replay_buffer_occupancy = minimal_replay_buffer_occupancy

        self.replay_buffer = ReplayBuffer(capacity, env.observation_space, env.action_space)

    def sample_ac(self, q, exploration):
        if self.exploration_type == 'softmax':
            # Softmax exploration
            temperature = exploration
            assert temperature > 0, 'Softmax temperature be greater than 0.'
            ac = torch.multinomial((q / temperature).exp(), 1)
            t_ac = ac.data[0, 0]
        else:
            # In this case, self.exploration_type == 'epsilon'
            # Epsilon-greedy exploration
            epsilon = exploration
            assert epsilon <= 1, 'Epsilon must be no greater than 1.'
            if np.random.rand() < epsilon:
                t_ac = np.random.randint(self.model.n_actions)
            else:
                # Greedy action
                max_q, ac = q.max(1)
                t_ac = ac.data[0]
        return t_ac

    def train_for(self, max_ticks, update_interval=1, batch_size=32, episode_report_interval=1, step_report_interval=1, checkpoint_interval=100, render=False):
        '''
        Args:
            max_ticks : int
                The number of ticks to sample from `self.env`.
            update_interval : int
                The maximum number of ticks to sample from `self.env` before updating the parameters.
            batch_size : int
                The number of (s, a, r, s') transitions to draw from replay buffer to use in an update. The expected usage of a transition before being discarded is lower bounded by `batch_size` / `update_interval`.
        '''
        done, t, step, episode = True, 0, 0, 0
        while t < max_ticks:
            # Reset the environment as needed
            if done:
                # Report the concluded episode
                if t > 0 and episode % episode_report_interval == 0:
                    logger.info('Episode %i length %i return %g', episode, episode_length, episode_return)
                    if self.writer is not None:
                        write_tb_event(self.writer, episode, {
                            'episodic/length': episode_length,
                            'episodic/return': episode_return,
                        })

                        write_tb_event(self.writer, t, {
                            'metrics/episode_return': episode_return,
                        })

                # Start a new episode
                ob = self.env.reset()
                done = False
                episode_return, episode_length = 0, 0
                episode += 1

            # Interact and generate data
            for i in xrange(update_interval):
                v_ob = Variable(torch.FloatTensor([ob.astype('float')]))
                q = self.model.q(v_ob)
                epsilon = linear_schedule(self.initial_exploration, self.terminal_exploration, 0, self.exploration_length, t)
                t_ac = self.sample_ac(q, epsilon)
                prev_ob = ob
                ob, r, done, extra = self.env.step(t_ac)
                t += 1
                episode_length += 1
                if render:
                    self.env.render()
                episode_return += r
                # Add (s, a, r, s') to replay buffer
                self.replay_buffer.add(prev_ob, t_ac, r, ob, done)

                if done:
                    break

            # Start training after accumulating some data
            if self.minimal_replay_buffer_occupancy < self.replay_buffer.occupancy:
                # Update parameters
                self.optimizer.zero_grad()

                # Sample from replay buffer
                obs, acs, rs, next_obs, dones = self.replay_buffer.sample_sars(batch_size)

                # Action-value estimation
                # TD(0)-error = Q(s, a) - (r + V(s')) where V(s') = max_a' Q(s', a')
                v_acs = Variable(torch.from_numpy(acs)).view(-1, 1)
                qs = self.model.q(Variable(torch.from_numpy(obs).float()))
                ac_qs = qs.gather(1, v_acs)
                # Copy model to target model
                if step % self.target_update_interval == 0:
                    copy_params(self.model, self.target_model)
                nonterminals = Variable(1 - torch.from_numpy(dones.astype('float')).float())
                vas = nonterminals * self.target_model.va(Variable(torch.from_numpy(next_obs).float()))
                target_q = Variable(torch.FloatTensor(rs)) + vas
                q_loss = self.q_crit(ac_qs, target_q.detach())

                # Total objective function
                loss = q_loss

                # Save a checkpoint
                if self.log_dir is not None and step % checkpoint_interval == 0:
                    save_checkpoint(t, episode, step, self.optimizer, self.model, self.log_dir)

                # Compute gradient
                loss.backward()

                # Clip the gradient
                if self.max_grad_norm > 0:
                    clip_grad_norm(self.model.parameters(), self.max_grad_norm)

                # Report model statistics
                if step % step_report_interval == 0:
                    # Report norms of parameters and gradient
                    param_norm = global_norm(self.model.parameters())
                    grad_norm = global_norm([param.grad for param in self.model.parameters() if param.grad is not None])

                    logger.info('Step %i total_loss %g value_loss %g epsilon %g grad_norm %g param_norm %g', step, loss.data[0], q_loss.data[0], epsilon, grad_norm.data[0], param_norm.data[0])
                    if self.writer is not None:
                        write_tb_event(self.writer, t, {
                            'train_loss/total_loss': loss.data[0],
                            'train_loss/value_loss': q_loss.data[0],
                            'train_extra/grad_norm': grad_norm.data[0],
                            'train_extra/param_norm': param_norm.data[0],
                            'train_extra/epsilon': epsilon,
                        })

                self.optimizer.step()
                step += 1

        return t, episode, step


# TODO add tabular Q model


@model_registry.register('dqn.mlp')
class DqnMlp(DqnModel):
    @classmethod
    def add_args(kls, parser, prefix):
        super().add_args(parser, prefix)
        parser.add_argument(kls.prefix_arg_name('layers', prefix), dest='hiddens', nargs='*', type=int, default=[64], help='Dimensionality of each hidden layers.')
        parser.add_argument(kls.prefix_arg_name('activation', prefix), dest='activation_fn', default='elu', choices=['elu', 'relu', 'sigmoid', 'tanh'], help='Non-linearity function to use after each linear layer.')

    def __init__(self, ob_space, ac_space, hiddens, activation_fn):
        super(DqnMlp, self).__init__()
        assert isinstance(ob_space, spaces.Box) and len(ob_space.shape) == 1, '`ob_space` can only support rank-1 `spaces.Box`.'
        assert isinstance(ac_space, spaces.Discrete), '`ac_space` can only support `spaces.Discrete`.'

        dim_ob = ob_space.shape[0]
        self.n_actions = ac_space.n
        widths = [dim_ob] + hiddens + [self.n_actions]
        self.fc_layers = []
        for i, (a, b) in enumerate(zip(widths[:-1], widths[1:])):
            lin = nn.Linear(a, b)
            self.add_module('fc%i' % i, lin)
            self.fc_layers.append(lin)
        self.activation_fn = getattr(f, activation_fn)

    def forward(self, ob):
        net = ob
        for fc in self.fc_layers[:-1]:
            net = fc(net)
            net = self.activation_fn(net)
        net = self.fc_layers[-1](net)
        return net
