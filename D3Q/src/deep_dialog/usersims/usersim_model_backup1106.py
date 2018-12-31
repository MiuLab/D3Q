from .usersim import UserSimulator
import argparse, json, random, copy, sys
import numpy as np
from model import ModelBasedUsersimulator
from deep_dialog import dialog_config


class ModelBasedSimulator(UserSimulator):
    """ A rule-based user simulator for testing dialog policy """
    def __init__(self, movie_dict=None, act_set=None, slot_set=None, start_set=None, params=None):
        """ Constructor shared by all user simulators """

        self.movie_dict = movie_dict
        self.act_set = act_set
        self.slot_set = slot_set
        self.start_set = start_set

        self.act_cardinality = len(act_set.keys())
        self.slot_cardinality = len(slot_set.keys())

        self.feasible_actions = dialog_config.feasible_actions
        self.num_actions = len(self.feasible_actions)

        self.max_turn = params['max_turn'] + 4
        self.state_dimension = 2 * self.act_cardinality + 7 * self.slot_cardinality + 3 + self.max_turn

        self.max_turn = params['max_turn']
        self.slot_err_probability = params['slot_err_probability']
        self.slot_err_mode = params['slot_err_mode']
        self.intent_err_probability = params['intent_err_probability']

        self.simulator_run_mode = params['simulator_run_mode']
        self.simulator_act_level = params['simulator_act_level']

        self.learning_phase = params['learning_phase']

        print 'building user simulator model'

        self.model = ModelBasedUsersimulator(2*(2 * self.slot_cardinality + self.act_cardinality) + 2 * self.slot_cardinality, 80, self.act_cardinality, 2 * self.slot_cardinality)

        self.training_corpus_path = u'D:\code\DialoguePlaning\TC-Bot_model_based_simulator\src\deep_dialog\data\human_huamn_data_framed_agent_first_turn.json'
        self.training_corpus = json.load(open(self.training_corpus_path))
        self.training_goal_ids = json.load(open(u'D:\code\DialoguePlaning\TC-Bot_model_based_simulator\src\deep_dialog\data\\user_goals_ids.json'))

        assert len(self.training_goal_ids) == len(self.start_set['all'])

        self.id_goals = dict(zip(self.training_goal_ids, self.start_set['all']))
        for e in xrange(400):
            cost = 0
            for i in self.id_goals.items()[0:170]:
                if len(self.training_corpus[i[0]]) < 2 or len(i[1]) < 2:
                    continue
                self.last_user_representation = np.zeros((1, 2 * self.slot_cardinality + self.act_cardinality))
                self.prepare_user_goal_representation(i[1])
                x, y_diaact, y_slots =self.prepare_state_representation_per_step(self.training_corpus[i[0]])
                x = np.vstack(x)[:,None,:]
                y_diaact = np.vstack(y_diaact)
                y_slots = np.vstack(y_slots)
                cost += self.model.train(x, y_diaact, y_slots)
            print cost
        # sys.exit(1)
        self.evalute_simulator_on_validation()

    def evalute_simulator_on_validation(self):

        acc_slot_all = 0
        c_slot_all = 0
        acc_diaact_all = 0
        c_diaact_all = 0

        slots_predict = []
        slots_groundtruth = []
        for i in self.id_goals.items()[170:]:
            if len(self.training_corpus[i[0]]) < 2 or len(i[1]) < 2:
                continue
            self.last_user_representation = np.zeros((1, 2 * self.slot_cardinality + self.act_cardinality))
            self.prepare_user_goal_representation(i[1])
            x, y_diaact, y_slots = self.prepare_state_representation_per_step(self.training_corpus[i[0]])
            x = np.vstack(x)[:, None, :]
            y_diaact = np.vstack(y_diaact)
            y_slots = np.vstack(y_slots)
            sample_diaact, prob_slots = self.model.train_sample(x,y_diaact,y_slots)

            slots_predict.append(prob_slots > 0.3)
            slots_groundtruth.append(y_slots)
            # for idx in range(sample_diaact.shape[0]):
            diaact_idx = np.argmax(sample_diaact,axis=1)
            # acc = (y_slots[0] == (prob_slots > 0.1)) * (y_slots[0]  == 1)
            # acc = acc.sum()
            # num_slots = np.sum(y_slots)
            # if num_slots > 0:
            #     acc_slot_all += acc
            #     c_slot_all += num_slots
            acc_diaact_all += sum(diaact_idx == np.argmax(y_diaact, axis=1))
            c_diaact_all += x.shape[0]
        print acc_slot_all, c_slot_all, acc_diaact_all, c_diaact_all
        slots_predict = np.vstack(slots_predict)
        slots_groundtruth = np.vstack(slots_groundtruth)

        for i in xrange(slots_predict.shape[1]):

            acc = slots_predict[:,i].astype(np.int32) & slots_groundtruth[:,i].astype(np.int32)
            acc = acc.sum()
            print acc,
            if slots_predict[:,i].sum() != 0:
                print float(acc) / slots_predict[:,i].sum(), acc /  slots_groundtruth[:,i].sum()
            else:
                print 0, acc /  slots_groundtruth[:,i].sum()

        sys.exit(1)

    def prepare_user_goal_representation(self, user_goal):

        request_slots_rep = np.zeros((1, self.slot_cardinality))
        inform_slots_rep = np.zeros((1, self.slot_cardinality))
        for s in user_goal['request_slots']:
            s = s.strip()
            request_slots_rep[0,self.slot_set[s]] = 1
        for s in user_goal['inform_slots']:
            s = s.strip()
            inform_slots_rep[0,self.slot_set[s]] = 1
        self.user_goal_representation = np.hstack([request_slots_rep, inform_slots_rep])

        return self.user_goal_representation

    def prepare_state_representation_per_step(self, message):
        # by default the first utterance is issued from agent side, e.g. say greeting
        # the sendond one is from user side saying moviename, need to refine the dialgoue sessions to add this two utterance

        train_x, train_y_diaact, train_y_slots = [],[],[]
        message = sorted(message.items(), key=lambda k:int(k[0]))
        for m_idx,m in enumerate(message):
            #m is a turn-level message
            m_message = m[1]['message']
            act_rep = np.zeros((1, self.act_cardinality))
            request_slots_rep = np.zeros((1, self.slot_cardinality))
            inform_slots_rep = np.zeros((1, self.slot_cardinality))
            for mm in m_message:
                diaact = mm[0].strip()
                act_rep[0, self.act_set[diaact]] = 1.0
                # if diaact.strip() != 'request' and diaact != 'inform':
                #     continue
                for slots in mm[1]:
                    slot = slots[0].strip()
                    if slot == '':
                        continue
                    if len(slots) == 1:
                        request_slots_rep[0, self.slot_set[slot]] = 1
                    else:
                        act_rep[0, self.act_set['inform']] = 1.0
                        inform_slots_rep[0, self.slot_set[slot]] = 1

                # print request_slots_rep.sum(), inform_slots_rep.sum()

            message_representation = np.hstack([act_rep,request_slots_rep,inform_slots_rep])
            actor = m[1]['actor']
            if actor == 'agent' and m_idx != len(message) - 1:
                message_representation = np.hstack([message_representation, self.last_user_representation, self.user_goal_representation])
                train_x.append(message_representation)
            if actor == 'user':
                self.last_user_representation = message_representation
                assert self.last_user_representation.shape[1] == 69
                # y.append(message_representation)
                train_y_diaact.append(act_rep)
                train_y_slots.append(np.hstack([request_slots_rep, inform_slots_rep]))

        return train_x, train_y_diaact, train_y_slots


    def initialize_episode(self):
        """ Initialize a new episode (dialog)
        state['history_slots']: keeps all the informed_slots
        state['rest_slots']: keep all the slots (which is still in the stack yet)
        """

        self.state = {}
        self.state['history_slots'] = {}
        self.state['inform_slots'] = {}
        self.state['request_slots'] = {}
        self.state['rest_slots'] = []
        self.state['turn'] = 0

        self.episode_over = False
        self.dialog_status = dialog_config.NO_OUTCOME_YET

        # self.goal =  random.choice(self.start_set)
        self.goal = self._sample_goal(self.start_set)
        self.goal['request_slots']['ticket'] = 'UNK'
        self.constraint_check = dialog_config.CONSTRAINT_CHECK_FAILURE

        """ Debug: build a fake goal mannually """
        # self.debug_falk_goal()

        # sample first action
        user_action = self._sample_action()

        self.last_user_action = None

        assert (self.episode_over != 1), ' but we just started'
        return user_action

    def _sample_goal(self, goal_set):
        """ sample a user goal  """

        sample_goal = random.choice(self.start_set[self.learning_phase])
        return sample_goal

    def prepare_state_representation(self, state):
        """ Create the representation for each state """

        user_action = state['user_action']
        current_slots = state['current_slots']
        kb_results_dict = state['kb_results_dict']
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

        turn_rep = np.zeros((1, 1)) + state['turn'] / 10.

        ########################################################################
        #  One-hot representation of the turn count?
        ########################################################################
        turn_onehot_rep = np.zeros((1, self.max_turn))
        turn_onehot_rep[0, state['turn']] = 1.0

        ########################################################################
        #   Representation of KB results (scaled counts)
        ########################################################################
        kb_count_rep = np.zeros((1, self.slot_cardinality + 1)) + kb_results_dict['matching_all_constraints'] / 100.
        for slot in kb_results_dict:
            if slot in self.slot_set:
                kb_count_rep[0, self.slot_set[slot]] = kb_results_dict[slot] / 100.

        ########################################################################
        #   Representation of KB results (binary)
        ########################################################################
        kb_binary_rep = np.zeros((1, self.slot_cardinality + 1)) + np.sum(
            kb_results_dict['matching_all_constraints'] > 0.)
        for slot in kb_results_dict:
            if slot in self.slot_set:
                kb_binary_rep[0, self.slot_set[slot]] = np.sum(kb_results_dict[slot] > 0.)

        self.final_representation = np.hstack(
            [user_act_rep, user_inform_slots_rep, user_request_slots_rep, agent_act_rep, agent_inform_slots_rep,
             agent_request_slots_rep, current_slots_rep, turn_rep, turn_onehot_rep, kb_binary_rep, kb_count_rep])
        return self.final_representation
