'''
created on Mar 08, 2018
@author: Shang-Yu Su
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
import torch.optim as optim
import numpy as np

use_cuda = torch.cuda.is_available()

class network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(network, self).__init__()
        self.model = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, output_size))

    def forward(self, inputs, testing=False):
        return self.model(inputs)


class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()

        # model
        self.model = network(input_size, hidden_size, output_size)
        # target model
        self.target_model = network(input_size, hidden_size, output_size)
        # first sync
        self.target_model.load_state_dict(self.model.state_dict())

        # hyper parameters
        self.gamma = 0.9
        self.reg_l2 = 1e-3
        self.max_norm = 1
        self.target_update_period = 100
        lr = 0.001

        self.optimizer = optim.RMSprop(self.model.parameters(), lr=lr)
        self.batch_count = 0

        # self.to(device)
        if use_cuda:
            self.cuda()

    def update_fixed_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def Variable(self, x):
        return Variable(x, requires_grad=False).cuda() if use_cuda else Variable(x, requires_grad=False)

    def singleBatch(self, batch):
        self.optimizer.zero_grad()
        loss = 0

        # each example in a batch: [s, a, r, s_prime, term]
        s = self.Variable(torch.FloatTensor(batch[0]))
        a = self.Variable(torch.LongTensor(batch[1]))
        r = self.Variable(torch.FloatTensor([batch[2]]))
        s_prime = self.Variable(torch.FloatTensor(batch[3]))

        q = self.model(s)
        q_prime = self.target_model(s_prime)

        # the batch style of (td_error = r + self.gamma * torch.max(q_prime) - q[a])
        td_error = r.squeeze_(0) + torch.mul(torch.max(q_prime, 1)[0], self.gamma).unsqueeze(1) - torch.gather(q, 1, a)
        loss += td_error.pow(2).sum()

        loss.backward()
        clip_grad_norm(self.model.parameters(), self.max_norm)
        self.optimizer.step()


    def predict(self, inputs):
        inputs = self.Variable(torch.from_numpy(inputs).float())
        return self.model(inputs, True).cpu().data.numpy()[0]

    def save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)
        print "model saved."

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        print "model loaded."
