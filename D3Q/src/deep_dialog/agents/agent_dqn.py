'''
Created on Jun 18, 2016

An DQN Agent

- An DQN
- Keep an experience_replay pool: training_data <State_t, Action, Reward, State_t+1>
- Keep a copy DQN

Command: python .\run.py --agt 9 --usr 1 --max_turn 40 --movie_kb_path .\deep_dialog\data\movie_kb.1k.json --dqn_hidden_size 80 --experience_replay_pool_size 1000 --replacement_steps 50 --per_train_epochs 100 --episodes 200 --err_method 2


@author: xiul
'''
import random, copy, json
import cPickle as pickle
import numpy as np

from deep_dialog import dialog_config
from agent import Agent
from deep_dialog.qlearning import DQN


class AgentDQN(Agent):
    def __init__(self, movie_dict=None, act_set=None, slot_set=None, params=None):
        self.movie_dict = movie_dict
        self.act_set = act_set
        self.slot_set = slot_set
        self.act_cardinality = len(act_set.keys())
        self.slot_cardinality = len(slot_set.keys())

        self.feasible_actions = dialog_config.feasible_actions
        self.num_actions = len(self.feasible_actions)

        self.epsilon = params['epsilon']
        self.agent_run_mode = params['agent_run_mode']
        self.agent_act_level = params['agent_act_level']
        self.experience_replay_pool = [] #experience replay pool <s_t, a_t, r_t, s_t+1>
        self.experience_replay_pool_from_model = []  # experience replay pool <s_t, a_t, r_t, s_t+1>

        self.experience_replay_pool_size = params.get('experience_replay_pool_size', 1000)
        self.hidden_size = params.get('dqn_hidden_size', 60)
        self.gamma = params.get('gamma', 0.9)
        self.predict_mode = params.get('predict_mode', False)
        self.warm_start = params.get('warm_start', 0)

        self.max_turn = params['max_turn'] + 5

        self.refine_state = True
        self.state_dimension = 2 * self.act_cardinality + 7 * self.slot_cardinality + 3 + self.max_turn
        if self.refine_state:
            self.state_dimension = 213

        self.dqn = DQN(self.state_dimension, self.hidden_size, self.num_actions)
        self.clone_dqn = copy.deepcopy(self.dqn)


        self.small_buffer = False
        self.cur_bellman_err = 0

        # replay buffer settings
        self.model_type = params['model_type']
        self.size_unit = params['buffer_size_unit']
        self.planning_steps = params['planning_steps']
        if params['planning_step_to_buffer']:
            if self.model_type == "DQN":
                self.max_user_buffer_size = self.size_unit * (self.planning_steps+1)
                self.max_world_model_buffer_size = 0
            else:
                # DDQ, D3Q
                self.max_user_buffer_size = self.size_unit
                self.max_world_model_buffer_size = self.size_unit * self.planning_steps
        else:
            if self.model_type == "DQN":
                self.max_user_buffer_size = self.size_unit
                self.max_world_model_buffer_size = 0
            else:
                # DDQ, D3Q
                self.max_user_buffer_size = self.size_unit
                self.max_world_model_buffer_size = self.size_unit
        '''
        # Prediction Mode: load trained DQN model
        if params['trained_model_path'] != None:
            # self.dqn.model = copy.deepcopy(self.load_trained_DQN(params['trained_model_path']))
            # self.clone_dqn = copy.deepcopy(self.dqn)
            self.dqn.load(params['trained_model_path'])
            self.predict_mode = True
            self.warm_start = 2
        '''
        self.available_actions = range(self.num_actions)
        self.new_actions = range(self.num_actions)

    def set_actions(self, the_actions):
        self.available_actions = copy.deepcopy(the_actions)
        self.new_actions = copy.deepcopy(the_actions)

    def add_actions(self, new_actions):
        self.new_actions = copy.deepcopy(new_actions)
        self.available_actions += new_actions
        # self.q_network.add_actions(new_actions)

    def initialize_episode(self):
        """ Initialize a new episode. This function is called every time a new episode is run. """

        self.current_slot_id = 0
        self.phase = 0
        self.request_set = ['moviename', 'starttime', 'city', 'date', 'theater', 'numberofpeople']

    def state_to_action(self, state):
        """ DQN: Input state, output action """
        # self.state['turn'] += 2
        self.representation = self.prepare_state_representation(state)
        self.action = self.run_policy(self.representation)
        if self.warm_start == 1:
            act_slot_response = copy.deepcopy(self.feasible_actions[self.action])
        else:
            act_slot_response = copy.deepcopy(self.feasible_actions[self.action])

        return {'act_slot_response': act_slot_response, 'act_slot_value_response': None}

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

        # turn_rep = np.zeros((1,1)) + state['turn'] / 10.
        turn_rep = np.zeros((1, 1))

        ########################################################################
        #  One-hot representation of the turn count?
        ########################################################################
        turn_onehot_rep = np.zeros((1, self.max_turn))
        turn_onehot_rep[0, state['turn']] = 1.0

        # ########################################################################
        # #   Representation of KB results (scaled counts)
        # ########################################################################
        # kb_count_rep = np.zeros((1, self.slot_cardinality + 1)) + kb_results_dict['matching_all_constraints'] / 100.
        # for slot in kb_results_dict:
        #    if slot in self.slot_set:
        #        kb_count_rep[0, self.slot_set[slot]] = kb_results_dict[slot] / 100.
        #
        # ########################################################################
        # #   Representation of KB results (binary)
        # ########################################################################
        # kb_binary_rep = np.zeros((1, self.slot_cardinality + 1)) + np.sum( kb_results_dict['matching_all_constraints'] > 0.)
        # for slot in kb_results_dict:
        #    if slot in self.slot_set:
        #        kb_binary_rep[0, self.slot_set[slot]] = np.sum( kb_results_dict[slot] > 0.)

        kb_count_rep = np.zeros((1, self.slot_cardinality + 1))
        ########################################################################
        #   Representation of KB results (binary)
        ########################################################################
        kb_binary_rep = np.zeros((1, self.slot_cardinality + 1))


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

    def run_policy(self, representation):
        """ epsilon-greedy policy """

        if random.random() < self.epsilon:
            return random.choice(self.available_actions)
        else:
            if self.warm_start == 1:
                if len(self.experience_replay_pool) > self.experience_replay_pool_size:
                    self.warm_start = 2
                return self.rule_policy()
            else:
                return self.available_actions[
                        np.argmax(self.dqn.predict(representation)[self.available_actions])
                    ]

    def rule_policy(self):
        """ Rule Policy """

        if self.current_slot_id < len(self.request_set):
            slot = self.request_set[self.current_slot_id]
            self.current_slot_id += 1

            act_slot_response = {}
            act_slot_response['diaact'] = "request"
            act_slot_response['inform_slots'] = {}
            act_slot_response['request_slots'] = {slot: "UNK"}
        elif self.phase == 0:
            act_slot_response = {'diaact': "inform", 'inform_slots': {'taskcomplete': "PLACEHOLDER"}, 'request_slots': {} }
            self.phase += 1
        elif self.phase == 1:
            act_slot_response = {'diaact': "thanks", 'inform_slots': {}, 'request_slots': {} }

        return self.action_index(act_slot_response)

    def action_index(self, act_slot_response):
        """ Return the index of action """

        for (i, action) in enumerate(self.feasible_actions):
            if act_slot_response == action:
                return i
        print act_slot_response
        raise Exception("action index not found")
        return None


    def register_experience_replay_tuple(self, s_t, a_t, reward, s_tplus1, episode_over, st_user, from_model=False):
        """ Register feedback from the environment, to be stored as future training data """

        state_t_rep = self.prepare_state_representation(s_t)
        action_t = self.action
        reward_t = reward
        state_tplus1_rep = self.prepare_state_representation(s_tplus1)
        st_user = self.prepare_state_representation(s_tplus1)
        training_example = (state_t_rep, action_t, reward_t, state_tplus1_rep, episode_over, st_user)
        if self.predict_mode == False: # Training Mode
            if self.warm_start == 1:
                self.experience_replay_pool.append(training_example)
        else: # Prediction Mode
            if not from_model:
                self.experience_replay_pool.append(training_example)
            else:
                self.experience_replay_pool_from_model.append(training_example)

        if len(self.experience_replay_pool) > self.max_user_buffer_size:
            self.experience_replay_pool = self.experience_replay_pool[-self.max_user_buffer_size:]

        if len(self.experience_replay_pool_from_model) > self.max_world_model_buffer_size:
            self.experience_replay_pool_from_model = self.experience_replay_pool_from_model[-self.max_world_model_buffer_size:]
    # run over the whole replay buffer
    def train(self, batch_size=16, num_iter=1, controller=0, use_real_example=True):
        """ Train DQN with experience replay """
        self.cur_bellman_err = 0
        self.cur_bellman_err_planning = 0
        running_expereince_pool = self.experience_replay_pool + self.experience_replay_pool_from_model

        for iter in range(num_iter):
            for _ in range(len(running_expereince_pool) / (batch_size)):

                batch = [random.choice(running_expereince_pool) for i in xrange(batch_size)]
                np_batch = []
                for x in range(5):
                    v = []
                    for i in xrange(len(batch)):
                        v.append(batch[i][x])
                    np_batch.append(np.vstack(v))
                batch_struct = self.dqn.singleBatch(np_batch)

            if len(self.experience_replay_pool) != 0:
                print ("cur bellman err %.4f, experience replay pool %s, model replay pool %s, cur bellman err for planning %.4f" % (
                float(self.cur_bellman_err) / (len(self.experience_replay_pool) / (float(batch_size))),len(self.experience_replay_pool), len(self.experience_replay_pool_from_model), self.cur_bellman_err_planning))

    # train specific number of batches
    # def train(self, batch_size=16, num_iter=1, planning=False, controller=0, use_real_example=True):
    def train_one_iter(self, batch_size=16, num_batches=1, planning=False, controller=0, use_real_example=True):
        """ Train DQN with experience replay """
        self.cur_bellman_err = 0
        self.cur_bellman_err_planning = 0
        running_expereince_pool = self.experience_replay_pool + self.experience_replay_pool_from_model
        for _ in range(num_batches):
            batch = [random.choice(self.experience_replay_pool) for i in xrange(batch_size)]
            np_batch = []
            for x in range(5):
                v = []
                for i in xrange(len(batch)):
                    v.append(batch[i][x])
                np_batch.append(np.vstack(v))

            batch_struct = self.dqn.singleBatch(np_batch)
        if len(self.experience_replay_pool) != 0:
            print ("cur bellman err %.4f, experience replay pool %s, cur bellman err for planning %.4f" % (
                float(self.cur_bellman_err) / (len(self.experience_replay_pool) / (float(batch_size))),
                len(self.experience_replay_pool), self.cur_bellman_err_planning))

    ################################################################################
    #    Debug Functions
    ################################################################################
    def save_experience_replay_to_file(self, path):
        """ Save the experience replay pool to a file """

        try:
            pickle.dump(self.experience_replay_pool, open(path, "wb"))
            print 'saved model in %s' % (path, )
        except Exception, e:
            print 'Error: Writing model fails: %s' % (path, )
            print e

    def load_experience_replay_from_file(self, path):
        """ Load the experience replay pool from a file"""

        self.experience_replay_pool = pickle.load(open(path, 'rb'))


    def load_trained_DQN(self, path):
        """ Load the trained DQN from a file """

        trained_file = pickle.load(open(path, 'rb'))
        model = trained_file['model']

        print "trained DQN Parameters:", json.dumps(trained_file['params'], indent=2)
        return model

    def set_user_planning(self, user_planning):
        self.user_planning = user_planning


    def save_dqn(self, path):
        # return self.dqn.unzip()
        self.dqn.save_model(path)

    def load_dqn(self, params):
        self.dqn.load(params)
