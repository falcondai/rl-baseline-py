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
from rl_baseline.common import evaluate_policy
from rl_baseline.registry import method_registry, model_registry, optimizer_registry
from rl_baseline.util import global_norm, log_format, write_tb_event, linear_schedule, copy_params, report_perf


# Set up logger
undo_logger_setup()
logging.basicConfig(format=log_format)
logger = logging.getLogger()
# Set the logging level
logger.setLevel(logging.DEBUG)


class DqnModel(Policy, ActionValue, StateValue, nn.Module, Parsable):
    def __init__(self, ob_space, *args, **kwargs):
        super(DqnModel, self).__init__(*args, **kwargs)
        self.ob_space = ob_space

    def preprocess_obs(self, obs, gpu_id=None):
        '''
        Maps an environment's observations into `Variable` for use with the model's `forward` calls.

        Args:
            obs : [`self.ob_space`]
                A list of observations from environment.
            gpu_id : int or `None`
                Appropriate GPU id to move the `Variable` to. None for CPU.
        '''
        if isinstance(self.ob_space, spaces.Box):
            v_obs = Variable(torch.FloatTensor(np.asarray(obs, dtype='float')))
        elif isinstance(self.ob_space, spaces.Discrete):
            v_obs = Variable(torch.from_numpy(np.asarray(obs, dtype='long')))
        # else:
        #     raise TypeError('`ob` cannot be preprocessed.')
        if gpu_id is not None:
            v_obs = v_obs.cuda(gpu_id)
        return v_obs

    def q(self, v_obs):
        return self.forward(v_obs)

    def va(self, v_obs):
        q = self.q(v_obs)
        max_q, ac = q.max(1)
        return max_q

    def act(self, ob, gpu_id=None):
        '''
        Args:
            ob : `self.ob_space`
                A single observation from the observation space.
        '''
        v_ob = self.preprocess_obs(obs=[ob], gpu_id=gpu_id)
        q = self.q(v_ob)
        max_q, max_ac = q.max(1)
        return max_ac.data[0]


class ReplayBuffer(Parsable):
    '''A circular buffer for storing (s, a, r, s') tuples.'''
    @classmethod
    def add_args(kls, parser, prefix):
        parser.add_argument(
            kls.prefix_arg_name('capacity', prefix),
            dest='capacity',
            type=int,
            default=5000,
            help='Size of the replay buffer.')

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

        logger.debug('Replay buffer took %i bytes', self.obs.nbytes + self.next_obs.nbytes + self.acs.nbytes + self.rs.nbytes + self.dones.nbytes)

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


# TODO add Trainer base class
@method_registry.register('dqn')
class DqnTrainer(Parsable):
    '''Q-learning'''
    @classmethod
    def add_args(kls, parser, prefix):
        parser.add_argument(
            kls.prefix_arg_name('criterion', prefix),
            dest='criterion',
            choices=['l2', 'huber'],
            default='l2',
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
            kls.prefix_arg_name('exp-start', prefix),
            dest='exploration_start',
            type=int,
            default=0,
            help='Start of the linear decay schedule of the exloration factor.')
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
        parser.add_argument(
            kls.prefix_arg_name('double-dqn', prefix),
            dest='double_dqn',
            action='store_true',
            help='Use double DQN. See Hasselt et al (2015).')
        parser.add_argument(
            kls.prefix_arg_name('update-interval', prefix),
            dest='update_interval',
            type=int,
            default=1,
            help='Maximum number of ticks to collect before updating the parameters.',
        )
        parser.add_argument(
            kls.prefix_arg_name('batch-size', prefix),
            dest='batch_size',
            type=int,
            default=32,
            help="Number of (s, a, r, s') transitions to draw from replay buffer to use in an update. The expected usage of a transition before being discarded is lower bounded by `batch_size` / `update_interval`.",
        )
        parser.add_argument(
            kls.prefix_arg_name('gamma', prefix),
            dest='gamma',
            type=float,
            default=1,
            help='The future reward discount.',
        )
        # Add replay buffer's arguments
        ReplayBuffer.add_args(parser, prefix)

    def __init__(self,
                 env,
                 model,
                 target_model,
                 optimizer,
                 capacity,
                 criterion,
                 max_grad_norm,
                 target_update_interval,
                 exploration_type,
                 initial_exploration,
                 terminal_exploration,
                 exploration_start,
                 exploration_length,
                 minimal_replay_buffer_occupancy,
                 double_dqn,
                 update_interval,
                 batch_size,
                 gamma,
                 gpu_id=None,
                 eval_env=None,
                 eval_summary_writer=None,
                 train_summary_writer=None,
                 saver=None):
        assert isinstance(model, DqnModel), 'The model argument needs to be an instance of `DqnModel`.'
        assert criterion in ['l2', 'huber'], '`criterion` has to be one of {`l2`, `huber`}.'
        assert exploration_type in ['softmax', 'epsilon'], 'Only supports `softmax` and `epsilon`-greedy exploration strategies.'

        self.env = env
        self.model = model
        self.target_model = target_model
        self.optimizer = optimizer

        # Optional utilities for logging
        self.train_summary_writer = train_summary_writer
        self.eval_summary_writer = eval_summary_writer
        self.eval_env = eval_env
        self.saver = saver

        # GPU utility
        self.gpu_id = gpu_id
        self.to_model_device = lambda t : t.cuda(self.gpu_id) if self.gpu_id is not None else t.cpu()

        # Total tick count
        self.total_ticks = 0
        self.q_crit = nn.SmoothL1Loss(size_average=False) if criterion == 'huber' else nn.MSELoss(size_average=False)
        self.max_grad_norm = max_grad_norm
        self.exploration_type = exploration_type
        self.initial_exploration = initial_exploration
        self.terminal_exploration = terminal_exploration
        self.exploration_start = exploration_start
        self.exploration_length = exploration_length
        self.target_update_interval = target_update_interval
        self.minimal_replay_buffer_occupancy = minimal_replay_buffer_occupancy
        self.double_dqn = double_dqn
        self.batch_size = batch_size
        self.update_interval = update_interval
        self.gamma = gamma

        # Replay buffer
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

    def train_for(self,
                  max_ticks,
                  episode_report_interval=1,
                  step_report_interval=1,
                  eval_interval=1,
                  n_eval_episodes=10,
                  checkpoint_interval=100,
                  render=False):
        '''
        Args:
            max_ticks : int
                The number of ticks to sample from `self.env`.
        '''
        # Initialize target model to be the same as the current mode
        copy_params(self.model, self.target_model)
        done, t, step, episode = True, 0, 0, 0
        while t < max_ticks:
            # Reset the environment as needed
            if done:
                # Report the concluded episode
                if t > 0 and episode % episode_report_interval == 0:
                    logger.info('Episode %i length %i return %g', episode, episode_length, episode_return)
                    if self.train_summary_writer is not None:
                        write_tb_event(self.train_summary_writer, episode, {
                            'episodic/length': episode_length,
                            'episodic/return': episode_return,
                        })

                        write_tb_event(self.train_summary_writer, t, {
                            'metrics/episode_return': episode_return,
                        })

                # Start a new episode
                ob = self.env.reset()
                done = False
                episode_return, episode_length = 0, 0
                episode += 1

            # Interact and generate data
            for i in xrange(self.update_interval):
                v_ob = self.model.preprocess_obs([ob], self.gpu_id)
                q = self.model.q(v_ob)
                epsilon = linear_schedule(self.initial_exploration, self.terminal_exploration, self.exploration_start, self.exploration_length, t)
                t_ac = self.sample_ac(q, epsilon)
                prev_ob = ob
                ob, r, done, extra = self.env.step(t_ac)
                t += 1
                episode_length += 1
                if render:
                    self.env.render()
                episode_return += r
                # Add (s, a, r, s') to replay buffer
                # TODO save preprocessed obs instead of raw obs?
                self.replay_buffer.add(prev_ob, t_ac, r, ob, done)

                if done:
                    break

            # Start training after accumulating some data
            if self.minimal_replay_buffer_occupancy < self.replay_buffer.occupancy:
                # Update parameters
                self.optimizer.zero_grad()

                # Sample from replay buffer
                obs, acs, rs, next_obs, dones = self.replay_buffer.sample_sars(self.batch_size)

                # Action-value estimation
                # TD(0)-error = Q(s, a) - (r + gamma * V(s')) where V(s') = max_a' Q(s', a')
                v_acs = Variable(torch.from_numpy(acs)).view(-1, 1)
                v_acs = self.to_model_device(v_acs)
                v_obs = self.model.preprocess_obs(obs, self.gpu_id)
                qs = self.model.q(v_obs)
                # Q(s, a)
                ac_qs = qs.gather(1, v_acs).squeeze(-1)
                # Compute V(s') = max_a' Q(s', a')
                # Copy model to target model
                if step % self.target_update_interval == 0:
                    copy_params(self.model, self.target_model)
                v_next_obs = self.model.preprocess_obs(next_obs, self.gpu_id)
                if self.double_dqn:
                    # Double DQN from Hasselt et al. _Deep Reinforcement Learning with Double Q-learning_
                    _, max_acs = self.model.q(v_next_obs).max(1)
                    vas = self.target_model.q(v_next_obs).gather(1, max_acs.view(-1, 1)).squeeze(-1)
                else:
                    # Standard DQN with a target Q-network
                    vas = self.target_model.va(v_next_obs).squeeze(-1)
                nonterminals = Variable(1 - torch.from_numpy(dones.astype('float')).float())
                nonterminals = self.to_model_device(nonterminals)
                vas = nonterminals * vas
                v_rs = Variable(torch.from_numpy(rs)).float()
                v_rs = self.to_model_device(v_rs)
                # Compute r + gamma * V(s')
                target_q = v_rs + self.gamma * vas
                q_loss = self.q_crit(ac_qs, target_q.detach())

                # Total objective function
                loss = q_loss

                # Save a checkpoint
                if self.saver is not None and step % checkpoint_interval == 0:
                    self.saver.save_checkpoint(t, episode, step)

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

                    logger.info('Step %i total_loss %.2f value_loss %.2f epsilon %.2f grad_norm %.2f param_norm %.2f', step, loss.data[0], q_loss.data[0], epsilon, grad_norm.data[0], param_norm.data[0])
                    if self.train_summary_writer is not None:
                        write_tb_event(self.train_summary_writer, t, {
                            'train_loss/total_loss': loss.data[0],
                            'train_loss/value_loss': q_loss.data[0],
                            'train_extra/grad_norm': grad_norm.data[0],
                            'train_extra/param_norm': param_norm.data[0],
                            'train_extra/epsilon': epsilon,
                        })

                # Evaluate the model
                if eval_interval != 0 and self.eval_env is not None and step % eval_interval == 0:
                    rets, lens = evaluate_policy(self.eval_env, self.model, n_eval_episodes, gpu_id=self.gpu_id, render=False)
                    avg_ret = np.mean(rets)
                    logger.info('Step %i eval:', step)
                    report_perf(rets, lens)
                    if self.eval_summary_writer is not None:
                        write_tb_event(self.eval_summary_writer, t, {
                            'metrics/episode_return': avg_ret,
                        })

                self.optimizer.step()
                step += 1

        return t, episode, step


@model_registry.register('dqn.tab')
class DqnTab(DqnModel):
    @classmethod
    def add_args(kls, parser, prefix):
        parser.add_argument(
            kls.prefix_arg_name('init-q', prefix),
            dest='initial_q',
            type=float,
            default=0,
            help='Initial Q-values. Positive to encourage exploration. None for random initialization of Q-values.',
        )

    def __init__(self, ob_space, ac_space, initial_q):
        super(DqnTab, self).__init__(ob_space)
        assert isinstance(ob_space, spaces.Discrete), '`ob_space` can only support `spaces.Discrete`.'
        assert isinstance(ac_space, spaces.Discrete), '`ac_space` can only support `spaces.Discrete`.'

        self.n_actions = int(ac_space.n)
        self.n_states = int(ob_space.n)
        self.q_values = nn.Linear(int(self.n_states), int(self.n_actions), bias=None)
        if initial_q is not None:
            nn.init.constant(self.q_values.weight, initial_q)

    def preprocess_obs(self, obs, gpu_id=None):
        obs = np.asarray(obs, dtype='int')
        # One-hot encode `obs`
        oh_obs = torch.zeros(len(obs), self.n_states)
        oh_obs.scatter_(1, torch.from_numpy(obs).view(-1, 1), 1)
        oh_obs = Variable(oh_obs)
        if gpu_id is not None:
            oh_obs = oh_obs.cuda(gpu_id)
        return oh_obs

    def forward(self, v_obs):
        return self.q_values(v_obs)


@model_registry.register('dqn.mlp')
class DqnMlp(DqnModel):
    @classmethod
    def add_args(kls, parser, prefix):
        super().add_args(parser, prefix)
        parser.add_argument(
            kls.prefix_arg_name('layers', prefix),
            dest='hiddens',
            nargs='*',
            type=int,
            default=[64],
            help='Dimensionality of each hidden layers.')
        parser.add_argument(
            kls.prefix_arg_name('activation', prefix),
            dest='activation_fn',
            default='elu',
            choices=['elu', 'relu', 'sigmoid', 'tanh'],
            help='Non-linearity function to use after each linear layer.')

    def __init__(self, ob_space, ac_space, hiddens, activation_fn):
        super(DqnMlp, self).__init__(ob_space)
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

    def forward(self, v_obs):
        net = v_obs
        for fc in self.fc_layers[:-1]:
            net = fc(net)
            net = self.activation_fn(net)
        net = self.fc_layers[-1](net)
        return net


@model_registry.register('dqn.tiled-tab')
class DqnTiledTab(DqnTab):
    @classmethod
    def add_args(kls, parser, prefix):
        super().add_args(parser, prefix)
        parser.add_argument(
            kls.prefix_arg_name('tile-shape', prefix),
            dest='ob_grid_shape',
            type=int,
            nargs='+',
            default=[16],
            help='Shape of the divided observation space.',
        )

    def __init__(self, ob_space, ac_space, initial_q, ob_grid_shape):
        assert isinstance(ob_space, spaces.Box), '`ob_space` needs to be an instance of `spaces.Box`.'
        assert ob_space.shape[0] == len(ob_grid_shape), '`ob_grid_shape` must have the same length as `ob_space.shape`.'
        self.ob_grid_shape = np.asarray(ob_grid_shape)
        self.span = ob_space.high - ob_space.low
        self.original_ob_space = ob_space
        self.ob_grid = 1 / self.ob_grid_shape
        self.tiled_ob_space = spaces.Discrete(int(np.prod(ob_grid_shape)))
        self.offsets = np.cumprod(np.concatenate([ob_grid_shape[1:], [1]])[::-1])[::-1]
        super().__init__(self.tiled_ob_space, ac_space, initial_q)

    def tile_obs(self, obs):
        x = (np.asarray(obs) - np.expand_dims(self.original_ob_space.low, 0)) / np.expand_dims(self.span, 0)
        idx = np.floor(x * self.ob_grid_shape)
        y = np.sum(idx * self.offsets, axis=1, dtype='int')
        return y

    def preprocess_obs(self, obs, gpu_id=None):
        obs = self.tile_obs(obs)
        v_obs = Variable(torch.from_numpy(obs).long())
        if gpu_id is not None:
            v_obs = v_obs.cuda(gpu_id)
        return v_obs


# TODO rbf MLP


# TODO CNN model


@model_registry.register('dqn.deepmind')
class DqnDeepMindModel(DqnModel):
    '''The model architecture from Mnih et al. _Human-level control through deep reinforcement learning_. A consistent definition is given in the published code: https://github.com/deepmind/dqn/blob/master/dqn/convnet_atari3.lua'''
    def __init__(self, ob_space, ac_space):
        super().__init__(ob_space)

        assert isinstance(ac_space, spaces.Discrete), '`ac_space` has to be an instance of `spaces.Discrete`.'
        self.ob_width = 84
        self.n_prev_frames = 4
        self.n_actions = ac_space.n
        # Assume that the input is an 4 x 84 x 84 image (4 stacked grayscale frames)
        self.cnn0 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        # 20 x 20 x 32
        self.cnn1 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # 9 x 9 x 64
        self.cnn2 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # 7 x 7 x 64 = 3136
        self.n_pre_fc = 7 * 7 * 64
        self.fc0 = nn.Linear(self.n_pre_fc, 512)
        # 512
        self.fc1 = nn.Linear(512, self.n_actions)
        self.activation_fn = f.relu

    def forward(self, v_obs):
        # Normalize
        net = v_obs / 255.
        net = self.cnn0(net)
        net = self.activation_fn(net)
        net = self.cnn1(net)
        net = self.activation_fn(net)
        net = self.cnn2(net)
        net = self.activation_fn(net)
        net = net.view(-1, self.n_pre_fc)
        net = self.fc0(net)
        net = self.activation_fn(net)
        net = self.fc1(net)
        return net
