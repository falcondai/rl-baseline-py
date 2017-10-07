import logging

import numpy as np
from gym import spaces, undo_logger_setup
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as f

from rl_baseline.core import StochasticPolicy, StateValue
from rl_baseline.registry import method_registry, model_registry, optimizer_registry
from rl_baseline.util import global_norm, log_format, write_tb_event


# Set up logger
undo_logger_setup()
logging.basicConfig(format=log_format)
logger = logging.getLogger(__name__)
# Set the logging level
logger.setLevel(logging.DEBUG)


class A2CModel(StochasticPolicy, StateValue, nn.Module):
    def __init__(self, *args, **kwargs):
        super(A2CModel, self).__init__(*args, **kwargs)

    def pi_va(self, ob):
        raise NotImplementedError


@method_registry.register('a2c')
class A2CTrainer:
    '''Advantage actor-critic'''
    def __init__(self, env, model, optimizer, writer=None):
        assert isinstance(model, A2CModel), 'The model argument needs to be an instance of `A2CModel`.'

        self.env = env
        self.model = model
        self.optimizer = optimizer

        # Optional utilities for logging
        self.writer = writer

        # Total tick count
        self.total_ticks = 0
        self.va_crit = nn.MSELoss()

    def train_for(self, max_ticks, batch_size, episode_report_interval=1, step_report_interval=1, render=False):
        done, t, step, episode = True, 0, 0, 0
        while t < max_ticks:
            # Reset the environment as needed
            if done:
                # Report the concluded episode
                if t > 0 and episode % episode_report_interval == 0:
                    logger.info('Episode %i length %i return %g', episode, total_length, total_return)
                    if self.writer is not None:
                        write_tb_event(self.writer, episode, {
                            'episodic/length': total_length,
                            'episodic/return': total_return,
                        })

                # Start a new episode
                ob = self.env.reset()
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
                pr, va = self.model.pi_va(v_ob)
                prs.append(pr)
                vas.append(va)
                t_ac = self.model.sample_ac(pr)
                ob, r, done, extra = self.env.step(t_ac)
                t += 1
                total_length += 1
                acs.append(t_ac)
                if render:
                    self.env.render()
                rs.append(r)
                total_return += r
                # End the batch if the episode ends
                if done:
                    break

            # Update parameters
            self.optimizer.zero_grad()

            # Last observation
            va_to_go = Variable(torch.zeros(1)) if done else self.model.va(Variable(torch.FloatTensor([ob]))).view(-1)
            vas.append(va)
            # Advantages
            advs = (Variable(torch.FloatTensor(np.cumsum(rs[::-1])[::-1])) + va_to_go - torch.cat(vas[:-1]).view(-1)).detach()
            # Policy gradient
            ps = torch.cat(prs)
            v_acs = Variable(torch.LongTensor(acs)).view(-1, 1)
            ac_ps = torch.cat(prs).gather(1, v_acs)
            logps = ac_ps.log()
            ac_ent = - torch.sum(ps * ps.log(), 1)
            ac_ent = ac_ent.mean()
            pg_loss = -(logps * advs.view(-1, 1)).mean()

            # State value estimation
            v_vas = torch.cat(vas).view(-1)
            # va_loss = 0.5 * va_crit(Variable(torch.FloatTensor(rs)) + v_vas[1:], v_vas[:-1].detach())
            # va_to_go = 0. if done else va.view(-1).detach()
            va_loss = 0.5 * self.va_crit(v_vas[:-1], (Variable(torch.FloatTensor(np.cumsum(rs[::-1])[::-1])) + va_to_go).detach())

            # Total objective function
            loss = pg_loss + 0.5 * va_loss - 0.01 * ac_ent
            loss.backward()

            # Report model statistics
            if step % step_report_interval == 0:
                # Report norms of parameters and gradient
                param_norm = global_norm(self.model.parameters())
                grad_norm = global_norm([param.grad for param in self.model.parameters() if param.grad is not None])

                logger.info('Step %i total_loss %g pg_loss %g value_loss %g action_entropy %g grad_norm %g param_norm %g', step, loss.data[0], pg_loss.data[0], va_loss.data[0], np.exp(ac_ent.data[0]), grad_norm.data[0], param_norm.data[0],)
                if self.writer is not None:
                    write_tb_event(self.writer, step, {
                        'train_loss/total_loss': loss.data[0],
                        'train_loss/pg_loss': pg_loss.data[0],
                        'train_loss/value_loss': va_loss.data[0],
                        'train_loss/action_entropy': np.exp(ac_ent.data[0]),
                        'train_extra/grad_norm': grad_norm.data[0],
                        'train_extra/param_norm': param_norm.data[0],
                    })

            self.optimizer.step()
            step += 1


@model_registry.register('a2c.linear')
class A2CLinearModel(A2CModel):
    def __init__(self, ob_space, ac_space):
        super(A2CLinearModel, self).__init__()
        assert isinstance(ob_space, spaces.Box) and len(ob_space.shape) == 1, '`ob_space` can only support rank-1 `spaces.Box`.'
        assert isinstance(ac_space, spaces.Discrete), '`ac_space` can only support `spaces.Discrete`.'

        dim_ob = ob_space.shape[0]
        n_actions = ac_space.n
        self.fc1 = nn.Linear(dim_ob, n_actions)
        self.non_linearity = f.elu
        self.fc2 = nn.Linear(dim_ob, 1)
        self.epsilon = 1e-10
        self.temperature = 1

    def forward_shared(self, ob):
        net = ob
        return net

    def forward_pi(self, shared):
        ac_score = self.fc1(shared)
        ac_score /= self.temperature
        ac_prob = f.softmax(ac_score)
        ac_prob = torch.clamp(ac_prob, self.epsilon, 1 - self.epsilon)
        return ac_prob

    def forward_va(self, shared):
        va = self.fc2(shared)
        return va

    def forward(self, ob):
        net = self.forward_shared(ob)
        ac_prob, va = self.forward_pi(net), self.forward_va(net)
        return ac_prob, va

    def pi_va(self, ob):
        return self.forward(ob)

    def pi(self, ob):
        net = self.forward_shared(ob)
        ac_prob = self.forward_pi(net)
        return ac_prob

    def va(self, ob):
        net = self.forward_shared(ob)
        va = self.forward_va(net)
        return va

    def sample_ac(self, ac_prob):
        ac = torch.multinomial(ac_prob, 1)
        t_ac = ac.data[0, 0]
        return t_ac
