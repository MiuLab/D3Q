'''
created on Mar 13, 2018
@author: Shang-Yu Su (t-shsu)
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
import torch.optim as optim
import numpy as np
import random
from deep_dialog import dialog_config

use_cuda = torch.cuda.is_available()

class Discriminator(nn.Module):
    def __init__(self, input_size=100, hidden_size=128, output_size=1, nn_type="MLP", movie_dict=None, act_set=None, slot_set=None, start_set=None, params=None):
        super(Discriminator, self).__init__()

        #############################
        #       misc setting       #
        #############################
        self.movie_dict = movie_dict
        self.act_set = act_set
        self.slot_set = slot_set
        self.start_set = start_set

        self.act_cardinality = len(act_set.keys())
        self.slot_cardinality = len(slot_set.keys())

        self.feasible_actions = dialog_config.feasible_actions
        self.feasible_actions_users = dialog_config.feasible_actions_users
        self.num_actions = len(self.feasible_actions)
        self.num_actions_user = len(self.feasible_actions_users)

        self.max_turn = params['max_turn'] + 5
        self.state_dimension = 213
        self.hidden_size = hidden_size
        self.cell_state_dimension = 213
        self.nn_type = nn_type
        self.threshold_upperbound = 0.55
        self.threshold_lowerbound = 0.45

        #############################
        #       model setting       #
        #############################
        # (1) MLP discriminator (2) RNN discriminator
        # (3) RNN encoder -> MLP discriminator
        if nn_type == "MLP":
            self.model = nn.Sequential(nn.Linear(self.state_dimension, hidden_size), nn.ELU(), nn.Linear(hidden_size, output_size), nn.Sigmoid())
        elif nn_type == "RNN":
            self.transform_layer = nn.Linear(self.cell_state_dimension, hidden_size)
            self.model = nn.LSTM(126, hidden_size, 1, dropout=0.00, bidirectional=False)
            self.output_layer = nn.Sequential(nn.Linear(hidden_size, output_size), nn.Sigmoid())

        self.user_model_experience_pool = list()
        self.user_experience_pool = list()
        # hyperparameters
        self.max_norm = 1
        lr = 0.001

        if nn_type == "MLP":
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=lr)
        elif nn_type == "RNN":
            params = []
            params.extend(list(self.transform_layer.parameters()))
            params.extend(list(self.model.parameters()))
            params.extend(list(self.output_layer.parameters()))
            self.optimizer = optim.RMSprop(params, lr=lr)
        self.BCELoss = nn.BCELoss()
        if use_cuda:
            self.cuda()

    def store_user_model_experience(self, experience):
        self.user_model_experience_pool.append(experience)
        if len(self.user_model_experience_pool) > 10000:
            self.user_model_experience_pool = self.user_model_experience_pool[-9000:]

    def store_user_experience(self, experience):
        self.user_experience_pool.append(experience)
        if len(self.user_experience_pool) > 10000:
            self.user_experience_pool = self.user_experience_pool[-9000:]

    def Variable(self, x):
        return Variable(x, requires_grad=False).cuda() if use_cuda else Variable(x, requires_grad=False)

    # discriminate a batch
    def forward(self, experience=[]):
        if self.nn_type == "MLP":
            # define the policy here
            d = [self.discriminate(exp).data.cpu().numpy()[0] for exp in experience]
            # NOTE: be careful
            if np.mean(d) < self.threshold_upperbound and np.mean(d) > self.threshold_lowerbound:
                return True
            else:
                return False
        elif self.nn_type == "RNN":
            # define the policy here
            d = [self.discriminate(exp).data.cpu().numpy()[0][0] for exp in experience]
            # NOTE: be careful
            if np.mean(d) < self.threshold_upperbound and np.mean(d) > self.threshold_lowerbound:
                return True
            else:
                return False

    def single_check(self, example):
        d = self.discriminate(example).data.cpu().numpy()[0]
        if d < self.threshold_upperbound and d > self.threshold_lowerbound:
            return True
        else:
            return False

    def discriminate(self, example):
        if self.nn_type == "MLP":
            state = self.prepare_state_representation(example[0])[0]
            model_input = self.Variable(torch.FloatTensor(state))
            return self.model(model_input)
        elif self.nn_type == "RNN":
            inputs = self.Variable(torch.FloatTensor([self.prepare_state_representation_for_RNN(history) for history in example[0]['history']]))
            h_0 = self.Variable(torch.FloatTensor(self.prepare_initial_state_for_RNN(example[0])))
            c_0 = self.Variable(torch.zeros(1, 1, self.hidden_size))
            output, hn = self.model(inputs, (self.transform_layer(h_0).unsqueeze(0), c_0))
            return self.output_layer(output[-1])

    # D(s, a) determines 'how real is the example'
    def train_single_batch(self, batch_size=16):
        self.optimizer.zero_grad()
        loss = 0
        # sample positive and negative examples
        pos_experiences = random.sample(self.user_experience_pool, batch_size)
        neg_experiences = random.sample(self.user_model_experience_pool, batch_size)

        for pos_exp, neg_exp in zip(pos_experiences, neg_experiences):
            loss += self.BCELoss(self.discriminate(pos_exp), self.Variable(torch.ones(1))) + self.BCELoss(self.discriminate(neg_exp), self.Variable(torch.zeros(1)))

        loss.backward()
        clip_grad_norm(self.parameters(), self.max_norm)
        self.optimizer.step()
        return loss

    def train(self, batch_size=16, batch_num=0):
        loss = 0
        if batch_num == 0:
            batch_num = min(len(self.user_experience_pool)/batch_size, len(self.user_model_experience_pool)/batch_size)

        for _ in range(batch_num):
            loss += self.train_single_batch(batch_size)
        return (loss.data.cpu().numpy()[0]/batch_num)

    def prepare_state_representation(self, state):
        """ Create the representation for each state """

        user_action = state['user_action']
        current_slots = state['current_slots']
        agent_last = state['agent_action']

        ########################################################################
        #   Create one-hot of acts to represent the current user action
        ########################################################################
        user_act_rep = np.zeros((1, self.act_cardinality))
        user_act_rep[0, self.act_set[user_action['diaact']]] = 1.0

        ########################################################################
        #   Create bag of inform slots representation to represent the current user action
        ########################################################################
        user_inform_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in user_action['inform_slots'].keys():
            user_inform_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Create bag of request slots representation to represent the current user action
        ########################################################################
        user_request_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in user_action['request_slots'].keys():
            user_request_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Creat bag of filled_in slots based on the current_slots
        ########################################################################
        current_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in current_slots['inform_slots']:
            current_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Encode last agent act
        ########################################################################
        agent_act_rep = np.zeros((1, self.act_cardinality))
        if agent_last:
            agent_act_rep[0, self.act_set[agent_last['diaact']]] = 1.0

        ########################################################################
        #   Encode last agent inform slots
        ########################################################################
        agent_inform_slots_rep = np.zeros((1, self.slot_cardinality))
        if agent_last:
            for slot in agent_last['inform_slots'].keys():
                agent_inform_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Encode last agent request slots
        ########################################################################
        agent_request_slots_rep = np.zeros((1, self.slot_cardinality))
        if agent_last:
            for slot in agent_last['request_slots'].keys():
                agent_request_slots_rep[0, self.slot_set[slot]] = 1.0

        # turn_rep = np.zeros((1, 1)) + state['turn'] / 10.
        turn_rep = np.zeros((1, 1))

        ########################################################################
        #  One-hot representation of the turn count?
        ########################################################################
        turn_onehot_rep = np.zeros((1, self.max_turn))
        turn_onehot_rep[0, state['turn']] = 1.0

        self.final_representation = np.hstack([user_act_rep, user_inform_slots_rep, user_request_slots_rep, agent_act_rep, agent_inform_slots_rep, agent_request_slots_rep, current_slots_rep, turn_rep, turn_onehot_rep])
        return self.final_representation

    def prepare_initial_state_for_RNN(self, state):
        user_action = state['user_action']
        current_slots = state['current_slots']
        agent_last = state['agent_action']

        ########################################################################
        #   Create one-hot of acts to represent the current user action
        ########################################################################
        user_act_rep = np.zeros((1, self.act_cardinality))
        user_act_rep[0, self.act_set[user_action['diaact']]] = 1.0

        ########################################################################
        #     Create bag of inform slots representation to represent the current user action
        ########################################################################
        user_inform_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in user_action['inform_slots'].keys():
            user_inform_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Create bag of request slots representation to represent the current user action
        ########################################################################
        user_request_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in user_action['request_slots'].keys():
            user_request_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Creat bag of filled_in slots based on the current_slots
        ########################################################################
        current_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in current_slots['inform_slots']:
            current_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Encode last agent act
        ########################################################################
        agent_act_rep = np.zeros((1, self.act_cardinality))
        if agent_last:
            agent_act_rep[0, self.act_set[agent_last['diaact']]] = 1.0

        ########################################################################
        #   Encode last agent inform slots
        ########################################################################
        agent_inform_slots_rep = np.zeros((1, self.slot_cardinality))
        if agent_last:
            for slot in agent_last['inform_slots'].keys():
                agent_inform_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Encode last agent request slots
        ########################################################################
        agent_request_slots_rep = np.zeros((1, self.slot_cardinality))
        if agent_last:
            for slot in agent_last['request_slots'].keys():
                agent_request_slots_rep[0, self.slot_set[slot]] = 1.0

        # turn_rep = np.zeros((1, 1)) + state['turn'] / 10.
        turn_rep = np.zeros((1, 1))

        ########################################################################
        #  One-hot representation of the turn count?
        ########################################################################
        turn_onehot_rep = np.zeros((1, self.max_turn))
        turn_onehot_rep[0, state['turn']] = 1.0

        self.final_representation = np.hstack([user_act_rep, user_inform_slots_rep, user_request_slots_rep, agent_act_rep, agent_inform_slots_rep, agent_request_slots_rep, current_slots_rep, turn_rep, turn_onehot_rep])
        return self.final_representation


    # {'request_slots': {'theater': 'UNK'}, 'turn': 0, 'speaker': 'user', 'inform_slots': {'numberofpeople': '3', 'moviename': '10 cloverfield lane'}, 'diaact': 'request'}
    def prepare_state_representation_for_RNN(self, state):

        ########################################################################
        #   Create one-hot of acts to represent the current user action
        ########################################################################
        user_act_rep = np.zeros((1, self.act_cardinality))
        if state['speaker'] == 'user':
            user_act_rep[0, self.act_set[state['diaact']]] = 1.0

        ########################################################################
        #   Create bag of inform slots representation to represent the current user action
        ########################################################################
        user_inform_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in state['inform_slots'].keys():
            user_inform_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Create bag of request slots representation to represent the current user action
        ########################################################################
        user_request_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in state['request_slots'].keys():
            user_request_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Encode last agent act
        ########################################################################
        agent_act_rep = np.zeros((1, self.act_cardinality))
        if state['speaker'] == 'agent':
            agent_act_rep[0, self.act_set[state['diaact']]] = 1.0

        turn_rep = np.zeros((1, 1))

        ########################################################################
        #  One-hot representation of the turn count?
        ########################################################################
        turn_onehot_rep = np.zeros((1, self.max_turn))
        turn_onehot_rep[0, state['turn']] = 1.0

        self.final_representation = np.hstack([user_act_rep, user_inform_slots_rep, user_request_slots_rep, agent_act_rep, turn_rep, turn_onehot_rep])

        return self.final_representation
