'''
created on Mar 12, 2018
@author: Shang-Yu Su (t-shsu)
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
import torch.optim as optim
import numpy as np

use_cuda = torch.cuda.is_available()

class SimulatorModel(nn.Module):
    def __init__(
            self,
            agent_action_size,
            hidden_size,
            state_size,
            user_action_size,
            reward_size=1,
            termination_size=1,
            nn_type="MLP",
            discriminator=None
    ):
        super(SimulatorModel, self).__init__()

        self.agent_action_size = agent_action_size
        self.nn_type = nn_type
        self.D = discriminator
        state_size = 270

        if nn_type == "MLP":
            self.s_enc_layer = nn.Linear(state_size, hidden_size)
            self.a_enc_layer = nn.Linear(agent_action_size, hidden_size)
            self.shared_layers = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.Tanh())
            self.au_pred_layer = nn.Sequential(nn.Linear(hidden_size, user_action_size), nn.LogSoftmax())
            self.s_next_pred_layer = nn.Linear(hidden_size, state_size)
            self.r_pred_layer = nn.Linear(hidden_size, reward_size)
            self.t_pred_layer = nn.Sequential(nn.Linear(hidden_size, termination_size), nn.Sigmoid())

        # hyper parameters
        self.max_norm = 1
        lr = 0.001

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.NLLLoss = nn.NLLLoss()
        self.MSELoss = nn.MSELoss()
        self.BCELoss = nn.BCELoss()

        if use_cuda:
            self.cuda()

    def Variable(self, x):
        return Variable(x, requires_grad=False).cuda() if use_cuda else Variable(x, requires_grad=False)

    # ex: [[2], [3], [42]]
    def one_hot(self, int_list, num_digits):
        int_list = np.array(int_list).squeeze()
        one_hot_list = np.eye(num_digits)[int_list]
        return one_hot_list

    def train(self, s_t, a_t, s_tp1, r_t, t_t, ua_t):
        if self.nn_type == "MLP":
            # [s_t, a_t, s_tp1, r_t, t_t, ua_t]
            self.optimizer.zero_grad()
            loss = 0

            s = self.Variable(torch.FloatTensor(s_t))
            a = self.Variable(torch.FloatTensor(self.one_hot(a_t, self.agent_action_size)))
            r = self.Variable(torch.FloatTensor(r_t))
            t = self.Variable(torch.FloatTensor(np.int32(t_t).squeeze()))
            au = self.Variable(torch.LongTensor(np.squeeze(ua_t)))

            h_s = self.s_enc_layer(s)
            h_a = self.a_enc_layer(a)

            h = self.shared_layers(torch.cat((h_s, h_a), 1))
            r_pred = self.r_pred_layer(h)
            t_pred = self.t_pred_layer(h)
            au_pred = self.au_pred_layer(h)

            # loss = self.NLLLoss(au_pred, au) + self.MSELoss(r_pred, r) + self.CrossEntropyLoss(t_pred, t)
            loss = self.NLLLoss(au_pred, au) + self.MSELoss(r_pred, r) + self.BCELoss(t_pred, t)
            loss.backward()
            clip_grad_norm(self.parameters(), self.max_norm)
            self.optimizer.step()
            return loss

    def predict(self, s, a):
        if self.nn_type == "MLP":
            s = self.Variable(torch.FloatTensor(s))
            a = self.Variable(torch.FloatTensor(self.one_hot(a, self.agent_action_size)))

            h_s = self.s_enc_layer(s)
            h_a = torch.unsqueeze(self.a_enc_layer(a), 0)
            h = self.shared_layers(torch.cat((h_s, h_a), 1))
            r_pred = self.r_pred_layer(h).cpu().data.numpy()
            t_pred = self.t_pred_layer(h).cpu().data.numpy()
            au_pred = torch.max(self.au_pred_layer(h), 1)[1].cpu().data.numpy()

            return au_pred, r_pred, t_pred


    def save_model(self, model_path):
        torch.save(self.state_dict(), model_path)
        print "model saved."

    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path))
        print "model loaded."

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

        self.final_representation = np.hstack(
            [
                user_act_rep,
                user_inform_slots_rep,
                user_request_slots_rep,
                agent_act_rep,
                agent_inform_slots_rep,
                agent_request_slots_rep,
                current_slots_rep,
                turn_rep,
                turn_onehot_rep
            ])
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

        self.final_representation = np.hstack([
            user_act_rep,
            user_inform_slots_rep,
            user_request_slots_rep,
            agent_act_rep,
            agent_inform_slots_rep,
            agent_request_slots_rep,
            current_slots_rep,
            turn_rep,
            turn_onehot_rep
        ])
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

        self.final_representation = np.hstack([
            user_act_rep,
            user_inform_slots_rep,
            user_request_slots_rep,
            agent_act_rep,
            turn_rep,
            turn_onehot_rep
        ])

        return self.final_representation
