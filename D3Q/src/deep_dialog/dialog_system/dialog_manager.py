"""
Created on May 17, 2016

@author: xiul, t-zalipt
"""

import json
from . import StateTracker
from deep_dialog import dialog_config
import copy


class DialogManager:
    """ A dialog manager to mediate the interaction between an agent and a customer """

    def __init__(self, agent, user, user_planning, act_set, slot_set, movie_dictionary, discriminator):
        self.agent = agent
        self.user = user
        self.user_planning = user_planning
        self.act_set = act_set
        self.slot_set = slot_set
        self.state_tracker = StateTracker(act_set, slot_set, movie_dictionary)
        self.user_action = None
        self.reward = 0
        self.episode_over = False

        self.user_actions_for_dump = []

        self.session_idx = 0
        self.use_model = False
        self.running_user = self.user

        self.discriminator = discriminator

    def initialize_episode(self, warm_start=False):
        """ Refresh state for new dialog """

        self.reward = 0
        self.episode_over = False
        self.state_tracker.initialize_episode()
        self.running_user = self.user
        self.use_model = False
        if not warm_start:
            self.running_user = self.user_planning
            self.use_model = True
        else:
            self.running_user = self.user
            self.use_model = False

        self.user_action = self.running_user.initialize_episode()

        if warm_start:
            self.user_planning.sample_goal = self.user.sample_goal

        self.state_tracker.update(user_action=self.user_action)

        if dialog_config.run_mode < 3:
            print ("New episode, user goal:")
            print json.dumps(self.user.goal, indent=2)
        self.print_function(user_action=self.user_action)

        self.agent.initialize_episode()

        # self.user_planning.set_user_goal(self.user.get_goal())

    def next_turn(self,
                  record_training_data=True,
                  record_training_data_for_user=True,
                  simulation_for_discriminator=False,
                  filter_experience_by_discriminator=False
                  ):
        """ This function initiates each subsequent exchange between agent and user (agent first) """

        ########################################################################
        #   CALL AGENT TO TAKE HER TURN
        ########################################################################
        self.state = self.state_tracker.get_state_for_agent()
        self.agent_action = self.agent.state_to_action(self.state)

        ########################################################################
        #   Register AGENT action with the state_tracker
        ########################################################################
        self.state_tracker.update(agent_action=self.agent_action)

        # ???
        self.state_user = self.state_tracker.get_state_for_user()

        self.agent.add_nl_to_action(self.agent_action) # add NL to Agent Dia_Act
        self.print_function(agent_action=self.agent_action['act_slot_response'])

        ########################################################################
        #   CALL USER TO TAKE HER TURN
        ########################################################################
        self.sys_action = self.state_tracker.dialog_history_dictionaries()[-1]
        if self.use_model:
            self.user_action, self.episode_over, self.reward = self.running_user.next(self.state_user, self.agent.action)
            # self.user_model_experience_pool.append((self.state_user, self.agent.action, ))
        else:
            self.user_action, self.episode_over, dialog_status = self.running_user.next(self.sys_action)
            self.reward = self.reward_function(dialog_status)

        ########################################################################
        #   Update state tracker with latest user action
        ########################################################################
        if self.episode_over != True:
            self.state_tracker.update(user_action=self.user_action)
            self.print_function(user_action=self.user_action)

        self.state_user_next = self.state_tracker.get_state_for_agent()

        # add into the pool
        if not simulation_for_discriminator:
            # store experiences for the discriminator
            if self.use_model:
                self.discriminator.store_user_model_experience((self.state_user, self.agent.action, self.state_user_next, self.reward, self.episode_over, self.user_action))
            else:
                self.discriminator.store_user_experience((self.state_user, self.agent.action, self.state_user_next, self.reward, self.episode_over, self.user_action))

            # store the experiences for the agent
            if self.use_model and filter_experience_by_discriminator:
                discriminate_check = self.discriminator.single_check((self.state, self.agent_action, self.reward, self.state_tracker.get_state_for_agent(), self.episode_over, self.state_user, self.use_model))
                if discriminate_check and record_training_data:
                    self.agent.register_experience_replay_tuple(self.state, self.agent_action, self.reward, self.state_tracker.get_state_for_agent(), self.episode_over, self.state_user, self.use_model)
            elif record_training_data:
                self.agent.register_experience_replay_tuple(self.state, self.agent_action, self.reward, self.state_tracker.get_state_for_agent(), self.episode_over, self.state_user, self.use_model)

            # store the experiences for the world model
            if record_training_data_for_user and not self.use_model:
                self.user_planning.register_experience_replay_tuple(self.state_user, self.agent.action, self.state_user_next, self.reward, self.episode_over, self.user_action)

            if self.use_model and filter_experience_by_discriminator:
                return (self.episode_over, self.reward, discriminate_check)
            else:
                return (self.episode_over, self.reward)
        else:
            return (self.state_user, self.agent.action, self.state_user_next, self.reward, self.episode_over, self.user_action)

    def reward_function(self, dialog_status):
        """ Reward Function 1: a reward function based on the dialog_status """
        if dialog_status == dialog_config.FAILED_DIALOG:
            reward = -self.user.max_turn #10
        elif dialog_status == dialog_config.SUCCESS_DIALOG:
            reward = 2*self.user.max_turn #20
        else:
            reward = -1
        return reward

    def reward_function_without_penalty(self, dialog_status):
        """ Reward Function 2: a reward function without penalty on per turn and failure dialog """
        if dialog_status == dialog_config.FAILED_DIALOG:
            reward = 0
        elif dialog_status == dialog_config.SUCCESS_DIALOG:
            reward = 2*self.user.max_turn
        else:
            reward = 0
        return reward


    def print_function(self, agent_action=None, user_action=None):
        """ Print Function """

        if agent_action:
            if dialog_config.run_mode == 0:
                if self.agent.__class__.__name__ != 'AgentCmd':
                    print ("Turn %d sys: %s" % (agent_action['turn'], agent_action['nl']))
            elif dialog_config.run_mode == 1:
                if self.agent.__class__.__name__ != 'AgentCmd':
                    print("Turn %d sys: %s, inform_slots: %s, request slots: %s" % (agent_action['turn'], agent_action['diaact'], agent_action['inform_slots'], agent_action['request_slots']))
            elif dialog_config.run_mode == 2: # debug mode
                print("Turn %d sys: %s, inform_slots: %s, request slots: %s" % (agent_action['turn'], agent_action['diaact'], agent_action['inform_slots'], agent_action['request_slots']))
                print ("Turn %d sys: %s" % (agent_action['turn'], agent_action['nl']))

            if dialog_config.auto_suggest == 1:
                print('(Suggested Values: %s)' % (self.state_tracker.get_suggest_slots_values(agent_action['request_slots'])))
        elif user_action:
            if dialog_config.run_mode == 0:
                print ("Turn %d usr: %s" % (user_action['turn'], user_action['nl']))
            elif dialog_config.run_mode == 1:
                print ("Turn %s usr: %s, inform_slots: %s, request_slots: %s" % (user_action['turn'], user_action['diaact'], user_action['inform_slots'], user_action['request_slots']))
            elif dialog_config.run_mode == 2: # debug mode, show both
                print ("Turn %d usr: %s, inform_slots: %s, request_slots: %s" % (user_action['turn'], user_action['diaact'], user_action['inform_slots'], user_action['request_slots']))
                print ("Turn %d usr: %s" % (user_action['turn'], user_action['nl']))

            if self.agent.__class__.__name__ == 'AgentCmd': # command line agent
                user_request_slots = user_action['request_slots']
                if 'ticket'in user_request_slots.keys(): del user_request_slots['ticket']
                if len(user_request_slots) > 0:
                    possible_values = self.state_tracker.get_suggest_slots_values(user_action['request_slots'])
                    for slot in possible_values.keys():
                        if len(possible_values[slot]) > 0:
                            print('(Suggested Values: %s: %s)' % (slot, possible_values[slot]))
                        elif len(possible_values[slot]) == 0:
                            print('(Suggested Values: there is no available %s)' % (slot))
                else:
                    kb_results = self.state_tracker.get_current_kb_results()
                    print ('(Number of movies in KB satisfying current constraints: %s)' % len(kb_results))

