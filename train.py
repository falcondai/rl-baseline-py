from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging, itertools
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
from torch.nn import functional as f

import gym
gym.undo_logger_setup()

from registry import env_registry, optimizer_registry
import registration

from util import log_format, global_norm, get_cartpole_state, set_cartpole_state

logging.basicConfig(format=log_format)
logger = logging.getLogger(__name__)

# Set the logging level
logger.setLevel(logging.DEBUG)
# Debug information
logger.debug('PyTorch version %s', torch.__version__)

class Policy(nn.Module):
    def __init__(self, dim_ob=4, n_actions=2):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(dim_ob, n_actions)
        # self.fc2 = nn.Linear(8, n_actions)
        self.non_linearity = f.elu
        self.fc2 = nn.Linear(dim_ob, 1)
        self.epsilon = 1e-10
        self.temperature = 1

    def forward(self, ob):
        net = ob
        ac_score, v = self.fc1(net), self.fc2(net)
        # net = self.non_linearity(net)
        # net = self.fc2(net)
        ac_score /= self.temperature
        ac_prob = f.softmax(ac_score)
        ac_prob = torch.clamp(ac_prob, self.epsilon, 1 - self.epsilon)
        return ac_prob, v

def random_cartpole_policy(ob):
    return 1 if np.random.rand() < 0.5 else 0

def q_from_rollout(sim, state, action, rollout_policy):
    # Start simulator at the current state
    ob = sim.reset()
    set_cartpole_state(sim, state)
    # And the current action
    ob, r, done, extra = sim.step(action)
    total_return = r
    while not done:
        a = rollout_policy(ob)
        ob, r, done, extra = sim.step(a)
        total_return += r
    return total_return

if __name__ == '__main__':
    import time, argparse, os

    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--environment', default='gym.CartPole-v0', help='Environment id.')
    # TODO add checkpoint save/restore
    parser.add_argument('-l', '--log-dir', default='/tmp/ail/t', help='Path to log directory.')
    parser.add_argument('--no-summary', dest='write_summary', action='store_false', help='Do not write summary protobuf for TensorBoard.')
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.1, help='Initial learning rate.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed.')
    parser.add_argument('--max-ticks', type=int, default=10**6, help='Maximum number of ticks to train.')
    parser.add_argument('-b', '--batch-size', type=int, default=20, help='Batch size.')
    # TODO add LR scheduler
    parser.add_argument('--lr-scheduler', default='none', help='Scheduler for learning rates.')
    # TODO add optimizer support
    parser.add_argument('-o', '--optimizer', default='sgd', help='Optimizer to use.')
    # TODO add GPU support
    parser.add_argument('--gpu', dest='use_gpu', action='store_true', help='Use GPU to compute.')

    args = parser.parse_args()

    if args.write_summary:
        try:
            # Tensorflow imports for writing summaries
            from tensorflow import summary, Summary
            from tensorflow.contrib.util import make_tensor_proto
            logger.debug('Imported TensorFlow.')
            # Summary writer and summary path
            summary_path = os.path.join(args.log_dir, '%i' % time.time())
            logger.info('Summary are written to %s' % summary_path)
            writer = summary.FileWriter(summary_path, flush_secs=10)
        except ImportError:
            logger.warn('TensorFlow cannot be imported. TensorBoard summaries will not be generated. Consider to install the CPU-version TensorFlow.')
            args.write_summary = False

    # Set up environment
    env_id = args.environment
    env = env_registry[env_id].make()
    logger.info('Environment id %s', env_id)
    logger.info('Environment observation space %r', env.observation_space)
    logger.info('Environment action space %r', env.action_space)
    logger.info('Environment reward range %r', env.reward_range)

    # Fix random seeds
    if args.seed is not None:
        logger.info('Set random seeds to %i' % args.seed)
        env.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Set up training methods
    pi = Policy()
    logger.info(pi)

    va_crit = torch.nn.MSELoss()

    # optimizer = optim.SGD(pi.parameters(), lr=0.001)
    optimizer = optim.Adam(pi.parameters(), lr=0.1)

    # max_search_samples = 200
    max_ticks = args.max_ticks

    # Upper bound of the batch size
    batch_size = args.batch_size
    t, done = 0, True
    episode = 0
    while t < max_ticks:
        # Reset the environment as needed
        if done:
            # Report the concluded episode
            if t > 0:
                logger.debug('Episodic return %g length %i', total_return, total_length)
                if args.write_summary:
                    summary_proto = Summary(value=[
                        Summary.Value(tag='episodic/total_length', simple_value=total_length),
                        Summary.Value(tag='episodic/total_return', simple_value=total_return),
                    ])
                    # Use the number of samples as a consistent measure regardless of batch sizes
                    writer.add_summary(summary_proto, global_step=episode)

            # Start a new episode
            ob = env.reset()
            done = False
            total_return, total_length = 0, 0
            episode += 1

        # SAR containers
        obs = []
        prs = []
        acs = []
        vas = []
        rs = []

        # Interact and generate data
        for i in range(batch_size):
            obs.append(ob)
            v_ob = Variable(torch.FloatTensor([ob]))
            pr, va = pi(v_ob)
            prs.append(pr)
            vas.append(va)
            # print(pr)
            ac = torch.multinomial(pr, 1)
            t_ac = ac.data[0, 0]
            ob, r, done, extra = env.step(t_ac)
            t += 1
            total_length += 1
            acs.append(t_ac)
            # env.render()
            rs.append(r)
            total_return += r
            # End the batch if the episode ends
            if done:
                break

        # Update parameters
        optimizer.zero_grad()

        # Last observation
        pr, va = pi(Variable(torch.FloatTensor([ob])))
        va_to_go = 0. if done else va.view(-1)
        vas.append(va)
        # Advantages
        advs = (Variable(torch.FloatTensor(np.cumsum(rs[::-1])[::-1])) + va_to_go - torch.cat(vas[:-1]).view(-1)).detach()
        # Policy gradient
        ps = torch.cat(prs)
        # ac_ps = torch.gather(torch.cat(prs), 1, Variable(torch.LongTensor(acs)))
        v_acs = torch.LongTensor(acs)
        ac_ps = torch.cat(prs)[:, v_acs]
        logps = ac_ps.log()
        ac_ent = - torch.sum(ps * ps.log(), 1)
        ac_ent = ac_ent.mean()
        pg_loss = -(logps * advs.view(-1, 1)).mean()

        # State value estimation
        v_vas = torch.cat(vas).view(-1)
        # va_loss = 0.5 * va_crit(Variable(torch.FloatTensor(rs)) + v_vas[1:], v_vas[:-1].detach())
        # va_to_go = 0. if done else va.view(-1).detach()
        va_loss = 0.5 * va_crit(v_vas[:-1], (Variable(torch.FloatTensor(np.cumsum(rs[::-1])[::-1])) + va_to_go).detach())

        # Total objective function
        loss = pg_loss + 0.5 * va_loss - 0.01 * ac_ent
        loss.backward()

        # Report norms of parameters and gradient
        if args.write_summary:
            param_norm = global_norm(pi.parameters())
            grad_norm = global_norm([param.grad for param in pi.parameters() if param.grad is not None])
            # grad_norm = torch.nn.utils.clip_grad_norm(pi.parameters(), 100)

            summary_proto = Summary(value=[
                Summary.Value(tag='train_loss/total_loss', simple_value=loss.data[0]),
                Summary.Value(tag='train_loss/pg_loss', simple_value=pg_loss.data[0]),
                Summary.Value(tag='train_loss/value_loss', simple_value=va_loss.data[0]),
                Summary.Value(tag='train_loss/action_entropy', simple_value=np.exp(ac_ent.data[0])),
                # Summary.Value(tag='train_extra/grad_norm', simple_value=grad_norm),
                Summary.Value(tag='train_extra/grad_norm', simple_value=grad_norm.data[0]),
                Summary.Value(tag='train_extra/param_norm', simple_value=param_norm.data[0]),
            ])
            writer.add_summary(summary_proto, global_step=t)

        optimizer.step()

        # # Choose an action by simulation (planning)
        # current_state = get_cartpole_state(env)
        # q = [0., 0.]
        # n_actions = len(q)
        # n_max = [0, 0]
        # sampled_qs = np.zeros(n_actions)
        # n_warmup_samples = 40
        # margin = 0.2
        # for n_samples in range(max_search_samples):
        #     for i, _ in enumerate(q):
        #         sampled_qs[i] = q_from_rollout(sim, current_state, i, random_cartpole_policy)
        #     a_max_q = np.argmax(sampled_qs)
        #     n_max[a_max_q] += 1
        #     # Test termination after warming up
        #     if n_samples > n_warmup_samples:
        #         if n_max[1] / n_samples - margin > n_max[0] / n_samples:
        #             a = 1
        #             break
        #         if n_max[0] / n_samples - margin > n_max[1] / n_samples:
        #             a = 0
        #             break
        # else:
        #     a = 1 if n_max[0] < n_max[1] else 0
        # print(n_samples, n_max)

    writer.close()
