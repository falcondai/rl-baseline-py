from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import numpy as np
from gym import spaces
import torch
from torch.autograd import Variable

from rl_baseline.registry import method_registry, model_registry
from rl_baseline.methods.dqn import DqnTab
from rl_baseline.core import Trainer
from rl_baseline.util import linear_schedule


@method_registry.register('td-q')
class QTrainer(Trainer):
    '''Backward view implementation of TD(lambda) Q-learning with eligibility traces.'''
    @classmethod
    def scaffold(kls, env_id, model_id, simulator_id=None):
        assert simulator_id == None, 'TD(lambda) does not use simulator.'

        from rl_baseline.registration import env_registry, model_registry
        kwargs = dict(
            env = env_registry[env_id].make(),
            model = model_registry[model_id],
        )
        return kwargs

    def __init__(self, env, model, td_lambda, gamma, exploration_type, initial_exploration, terminal_exploration, exploration_start, exploration_length, initial_step_size, step_size_schedule, eligibility_type, train_summary_writer=None, eval_summary_writer=None, eval_env=None, n_eval_episodes=0, gpu_id=None):
        # TODO relax this to non-discrete finite spaces
        assert isinstance(env.observation_space, spaces.Discrete), 'env.observation_space must be discrete.'
        assert isinstance(env.action_space, spaces.Discrete), 'env.action_space must be discrete.'
        assert exploration_type in ['softmax', 'epsilon'], 'Only supports `softmax` and `epsilon`-greedy exploration strategies.'
        assert eligibility_type in ['accumulating', 'replacing'], 'Only supports `accumulating` and `replacing` eligibility traces.'

        super().__init__(env=env, model=model, eval_env=eval_env, n_eval_episodes=n_eval_episodes, gpu_id=gpu_id)

        self.td_lambda = td_lambda
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        self.exploration_type = exploration_type
        self.gamma = gamma
        self.eligibility_type = eligibility_type
        self.initial_exploration = initial_exploration
        self.terminal_exploration = terminal_exploration
        self.exploration_start = exploration_start
        self.exploration_length = exploration_length
        self.initial_step_size = initial_step_size
        self.step_size_schedule = step_size_schedule
        # self.criterion = nn.MSELoss(size_average=False)

    def sample_exploratory_action(self, q, exploration):
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
                t_ac = np.random.randint(self.n_actions)
            else:
                # Greedy action
                # max_q, ac = q.max(1)
                # t_ac = ac.data[0]
                t_ac = np.argmax(q)
        return t_ac

    def sample_action(self, ob):
        v_obs = self.model.preprocess_obs([ob])
        q = self.model.q(v_obs)
        exploration = linear_schedule(self.initial_exploration, self.terminal_exploration, self.exploration_start, self.exploration_length, self.tick)
        return self.sample_exploratory_action(q, exploration)

    def transition_sampler(self):
        ob = self.env.reset()
        while True:
            # v_obs = self.model.preprocess_obs([ob])
            # q = self.model.q(v_obs)
            q = self.model[ob]
            exploration = linear_schedule(self.initial_exploration, self.terminal_exploration, self.exploration_start, self.exploration_length, self.tick)
            ac = self.sample_exploratory_action(q, exploration)
            next_ob, r, done, extra = self.env.step(ac)
            # Return (S, A, R, S') transitions
            extra['td.q'] = q
            yield ob, ac, r, next_ob, done, extra
            ob = self.env.reset() if done else next_ob

    def train_for(self, max_ticks):
        # Set eligibility traces to 0
        e = np.zeros((self.n_states, self.n_actions))
        for ob, ac, r, next_ob, done, extra in self.transition_sampler():
            self.update_stats(r, done)

            # Compute TD-error = (r + gamma * max_a' Q(s', a')) - Q(s, a)
            # v_obs = self.model.preprocess_obs([next_ob])
            # next_va = self.model.va(v_obs) if not done else 0
            next_va = self.model[ob].max() if not done else 0
            q_ac = extra['td.q'][ac]
            delta = (r + self.gamma * next_va) - q_ac
            # Update the eligibility trace of (s, a)
            if self.eligibility_type == 'accumulating':
                e[ob, ac] += 1
            elif self.eligibility_type == 'replacing':
                e[ob, ac] = 1
            if delta != 0:
                # Update action values
                alpha = self.step_size_schedule(self.tick)
                # TODO make this more efficient
                for s in xrange(self.n_states):
                    for a in xrange(self.n_actions):
                        self.model[s, a] += alpha * delta * e[s, a]
                        # Naive Q(lambda) according to Sutton & Barto p.184
                        e[s, a] *= self.td_lambda * self.gamma

            self.report_step(self.n_eval_episodes)
            self.step += 1

            if done:
                e = np.zeros((self.n_states, self.n_actions))

            # Stop
            if max_ticks < self.tick:
                break


@method_registry.register('td-sarsa')
class SarsaTrainer(Trainer):
    '''Backward view implementation of TD(lambda) Sarsa-learning with eligibility traces.'''
    @classmethod
    def scaffold(kls, env_id, model_id, simulator_id=None):
        assert simulator_id == None, 'TD(lambda) does not use simulator.'

        from rl_baseline.registration import env_registry, model_registry
        kwargs = dict(
            env = env_registry[env_id].make(),
            model = model_registry[model_id],
        )
        return kwargs

    def __init__(self, env, model, optimizer, td_lambda, gamma, exploration_type, initial_exploration, terminal_exploration, exploration_start, exploration_length, step_size_schedule, eligibility_type, train_summary_writer=None, eval_summary_writer=None, eval_env=None, n_eval_episodes=0, gpu_id=None):
        # TODO relax this to non-discrete finite spaces
        assert isinstance(env.observation_space, spaces.Discrete), 'env.observation_space must be discrete.'
        assert isinstance(env.action_space, spaces.Discrete), 'env.action_space must be discrete.'
        assert exploration_type in ['softmax', 'epsilon'], 'Only supports `softmax` and `epsilon`-greedy exploration strategies.'
        assert eligibility_type in ['accumulating', 'replacing'], 'Only supports `accumulating` and `replacing` eligibility traces.'

        super().__init__(env=env, model=model, optimizer=optimizer, eval_env=eval_env, n_eval_episodes=n_eval_episodes, gpu_id=gpu_id)

        self.td_lambda = td_lambda
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        self.exploration_type = exploration_type
        self.gamma = gamma
        self.eligibility_type = eligibility_type
        self.initial_exploration = initial_exploration
        self.terminal_exploration = terminal_exploration
        self.exploration_start = exploration_start
        self.exploration_length = exploration_length
        self.step_size_schedule = step_size_schedule

    def sample_exploratory_action(self, q, exploration):
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
                t_ac = np.random.randint(self.n_actions)
            else:
                # Greedy action
                max_q, ac = q.max(1)
                t_ac = ac.data[0]
        return t_ac

    def sample_action_q(self, ob):
        v_obs = self.model.preprocess_obs([ob])
        q = self.model.q(v_obs)
        exploration = linear_schedule(self.initial_exploration, self.terminal_exploration, self.exploration_start, self.exploration_length, self.tick)
        return self.sample_exploratory_action(q, exploration), q

    def train_for(self, max_ticks):
        # Set eligibility traces to 0
        e = [Variable(torch.zeros(param.size())) for param in self.model.parameters()]

        ob = self.env.reset()
        ac, q = self.sample_action_q(ob)
        while self.tick < max_ticks:
            next_ob, r, done, extra = self.env.step(ac)
            self.update_stats(r, done)
            next_ac, next_q = self.sample_action_q(next_ob)

            # Compute TD-error = (r + gamma * Q(s', a')) - Q(s, a)
            next_va = next_q[0, next_ac] if not done else 0
            q_ac = q[0, ac]
            delta = (r + self.gamma * next_va) - q_ac

            self.optimizer.zero_grad()
            # Compute grad Q(s, a)
            q_ac.backward()
            # Update action values
            alpha = self.step_size_schedule(self.tick)
            for ep, param in zip(e, self.model.parameters()):
                # Update the eligibility traces
                if self.eligibility_type == 'accumulating':
                    ep += param.grad.detach()
                else:
                    # In this case self.eligibility_type == 'replacing'
                    raise NotImplementedError
                param.grad = - alpha * delta.detach() * ep
                ep *= self.td_lambda * self.gamma
            self.optimizer.step()

            self.report_step(self.n_eval_episodes)
            self.step += 1

            if done:
                # Set eligibility traces to 0
                for ep in e:
                    ep.data.zero_()
                # Start another episode
                ob = self.env.reset()
                ac, q = self.sample_action_q(ob)
            else:
                ac, q = next_ac, next_q
                ob = next_ob

# TODO implement LSTDQ


# Register tabular models
model_registry.register_to(DqnTab, 'td-q.tab')
model_registry.register_to(DqnTab, 'td-sarsa.tab')
