from reward_machines.sparse_reward_machine import SparseRewardMachine
from tester.tester import Tester
import numpy as np
import random, time, os
import matplotlib.pyplot as plt

class IhrlAgent:
    """
    Class meant to represent an independent hierarchical agent.
    The agent maintains a representation of its own q-function and accumulated reward
    which are updated across training episodes.
    """
    def __init__(self, options_list, s_i, num_states, num_meta_states, actions, agent_id):
        """
        Initialize agent object.

        Parameters
        ----------
        rm_file : str
            File path pointing to the reward machine this agent is meant to use for learning.
        options_list : list
            list of strings describing the different options available to the agent
        s_i : int
            Index of initial state.
        num_states : int
            Number of states in the environment
        num_meta_states : int
            Number of meta states in the environment
        actions : list
            List of actions available to the agent.
        agent_id : int
            Index of this agent.
        """
        self.options_list = options_list
        self.agent_id = agent_id
        self.s_i = s_i
        self.s = s_i
        self.actions = actions
        self.num_states = num_states
        self.num_meta_states = num_meta_states

        self.meta_actions = np.arange(len(self.options_list))
        self.meta_q = np.zeros([num_meta_states, len(self.options_list)])

        self.q_dict = dict()
        for option in options_list:
            self.q_dict[option] = np.zeros((self.num_states, len(self.actions)))

        self.current_option = ''
        self.option_start_state = -1
        self.option_complete = False

    def reset_state(self):
        """
        Reset the agent to the initial state of the environment.
        """
        self.s = self.s_i

    def reset_option(self):
        """
        Reset the agent to have no currently active option.
        """
        self.current_option = ''
        self.option_complete = False

    def get_next_meta_action(self, meta_state, avail_meta_action_indeces, epsilon, learning_params):
        """
        Return the next meta-action selected by the agent.

        Outputs
        -------
        g_selected : int
            Selected next action for this agent.
        """
        T = learning_params.T

        if random.random() < epsilon:
            # g = random.choice(self.meta_actions)
            g = random.choice(avail_meta_action_indeces)
            g_selected = g
        else:
            pr_sum = np.sum(np.exp(self.meta_q[meta_state, :] * T))
            pr = np.exp(self.meta_q[meta_state, :] * T)/pr_sum # pr[g] is probability of taking option g

            # If any q-values are so large that the softmax function returns infinity, 
            # make the corresponding actions equally likely
            if any(np.isnan(pr)):
                print('BOLTZMANN CONSTANT TOO LARGE IN ACTION-SELECTION SOFTMAX.')
                temp = np.array(np.isnan(pr), dtype=float)
                pr = temp / np.sum(temp)

            pr_select = np.zeros([len(self.meta_actions) + 1, 1])
            pr_select[0] = 0
            for i in range(len(self.meta_actions)):
                pr_select[i+1] = pr_select[i] + pr[i]

            while True:
                randn = random.random()
                for g in self.meta_actions:
                    if randn >= pr_select[g] and randn <= pr_select[g+1]:
                        g_selected = g
                        break
                # Only choose allowable meta-actions
                if g_selected in avail_meta_action_indeces:
                    break

        return g_selected

    def get_next_action(self, epsilon, learning_params):
        """
        Return the action next action selected by the agent.

        Outputs
        -------
        a : int
            Selected next action for this agent.
        """
        T = learning_params.T
        option = self.current_option
        q = self.q_dict[option]

        if random.random() < epsilon:
            a = random.choice(self.actions)
            a_selected = a
        else:
            pr_sum = np.sum(np.exp(q[self.s, :] * T))
            pr = np.exp(q[self.s, :] * T)/pr_sum # pr[a] is probability of taking action a

            # If any q-values are so large that the softmax function returns infinity, 
            # make the corresponding actions equally likely
            if any(np.isnan(pr)):
                print('BOLTZMANN CONSTANT TOO LARGE IN ACTION-SELECTION SOFTMAX.')
                temp = np.array(np.isnan(pr), dtype=float)
                pr = temp / np.sum(temp)

            pr_select = np.zeros([len(self.actions) + 1, 1])
            pr_select[0] = 0
            for i in range(len(self.actions)):
                pr_select[i+1] = pr_select[i] + pr[i]

            randn = random.random()
            for a in self.actions:
                if randn >= pr_select[a] and randn <= pr_select[a+1]:
                    a_selected = a
                    break

        return a_selected

    def update_agent(self, s_new, avail_options, a, reward, completed_options, learning_params, update_q_function=True):
        """
        Update the agent's state, q-function, and reward machine after 
        interacting with the environment.

        Parameters
        ----------
        s_new : int
            Index of the agent's next state.
        a : int
            Action the agent took from the last state.
        reward : float
            Reward the agent achieved during this step.
        completed_options : list
            list of strings corresponding to the completed options.
        learning_params : LearningParameters object
            Object storing parameters to be used in learning.
        update_q_function : bool
            Boolean representing whether or not the q-function should be updated
        """
        
        if update_q_function == True:
            for option in self.options_list:
                if option in avail_options:
                    if option in completed_options:
                        reward = 1.0
                        self.update_q_function(self.s, s_new, a, option, reward, learning_params)
                    else:
                        reward = 0.0
                        self.update_q_function(self.s, s_new, a, option, reward, learning_params)

        # Moving to the next state
        self.s = s_new

    def update_q_function(self, s, s_new, a, option, reward, learning_params):
        """
        Update the q function using the action, states, and reward value.

        Parameters
        ----------
        s : array
            Indeces of the agents' previous state
        s_new : array
            Indeces of the agents' updated state
        a : int
            Index of low-level action taken
        option : string
            String describing the option whose q function is being updated.
        reward : float
            Intrinsic reward. Should be 1 if option was completed in moving from s to s_new, 0 otherwise.
        a : int
            Action the agent took from state s
        learning_params : LearningParameters object
            Object storing parameters to be used in learning.
        """
        alpha = learning_params.alpha
        gamma = learning_params.gamma

        # Bellman update
        self.q_dict[option][s, a] = (1-alpha)*self.q_dict[option][s, a] + alpha*(reward + gamma*np.amax(self.q_dict[option][s_new]))

    def update_meta_q_function(self, meta_state, meta_state_new, g, mc_reward, learning_params):
        """
        """
        alpha = learning_params.alpha
        gamma = learning_params.gamma

        discount_vec = np.zeros(len(mc_reward))
        discount_vec[0] = gamma
        for t in range(1, len(mc_reward)):
            discount_vec[t] = discount_vec[t-1] * gamma

        reward = np.dot(discount_vec, mc_reward)
        N = len(mc_reward)

        self.meta_q[meta_state][g] = (1-alpha)*self.meta_q[meta_state][g] + alpha*(reward + (gamma**N) *np.amax(self.meta_q[meta_state_new]))