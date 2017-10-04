from registration import env_registry, method_registry, optimizer_registry
import torch
from torch import nn
import torch.nn.functional as f
from methods.a2c import A2CModel

# Define your own model. As long as it inherits from the compatible model class, the desired trainer (in this case, A2C) can use it.
class MyA2CModel(A2CModel):
    def __init__(self, dim_ob=4, n_actions=2):
        super(MyA2CModel, self).__init__()
        hidden_dim = 2
        self.fc1 = nn.Linear(dim_ob, hidden_dim)
        self.pi_fc = nn.Linear(hidden_dim, n_actions)
        self.va_fc = nn.Linear(hidden_dim, 1)
        self.non_linearity = f.elu
        self.epsilon = 1e-10

    def forward(self, ob):
        net = ob
        net = self.fc1(net)
        net = self.non_linearity(net)
        ac_score, va = self.pi_fc(net), self.va_fc(net)
        ac_prob = f.softmax(ac_score)
        # Prevent zeros
        ac_prob = torch.clamp(ac_prob, self.epsilon, 1 - self.epsilon)
        return ac_prob, va

    def pi_va(self, ob):
        return self.forward(ob)

    def pi(self, ob):
        return self.forward(ob)[0]

    def va(self, ob):
        return self.forward(ob)[1]

    def sample_ac(self, ac_prob):
        return torch.multinomial(ac_prob, 1).data[0, 0]


from util import report_per_episode
from functools import partial

env = env_registry['gym.CartPole-v0'].make()
mod = MyA2CModel(4, 2)
print(mod)
opt = optimizer_registry['SGD'](params=mod.parameters(), lr=0.01)
tra = method_registry['a2c'](env, mod, opt, partial(report_per_episode, False, None, 1))

# Train for a little
tra.train_for(1000, 20, False)
