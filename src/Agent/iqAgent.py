from reward_machines.sparse_reward_machine import SparseRewardMachine
from tester.tester import Tester
import numpy as np
import random, time, os
import matplotlib.pyplot as plt

class iqAgent:
    """
    Class meant to represent an independent q-learning agent.
    The agent maintains a representation of its own q-function and accumulated reward
    which are updated across training episodes.
    
    Note: Users of this class must manually reset the world state and the meta_state
    state when starting a new episode by calling self.initialize_world()
    """
    def __init__(self, s_i, meta_state_i, num_states, num_meta_states, actions, agent_id):
        """
        Initialize agent object.

        Parameters
        ----------
        s_i : int
            Index of initial state.
        actions : list
            List of actions available to the agent.
        agent_id : int
            Index of this agent.
        """
        self.agent_id = agent_id
        self.s_i = s_i
        self.s = s_i
        self.meta_state_i = meta_state_i
        self.meta_state = meta_state_i
        self.actions = actions
        self.num_states = num_states
        self.num_meta_states = num_meta_states
        
        self.q = np.zeros([num_states, num_meta_states, len(self.actions)])
        self.total_local_reward = 0

    def reset_state(self):
        """
        Reset the agent to the initial state of the environment.
        """
        self.s = self.s_i
        self.meta_state = self.meta_state_i

    def get_next_action(self, epsilon, learning_params):
        """
        Return the action next action selected by the agent.

        Outputs
        -------
        s : int
            Index of the agent's current state.
        a : int
            Selected next action for this agent.
        """

        T = learning_params.T

        if random.random() < epsilon:
            a = random.choice(self.actions)
            a_selected = a
        else:
            pr_sum = np.sum(np.exp(self.q[self.s, self.meta_state, :] * T))
            pr = np.exp(self.q[self.s, self.meta_state, :] * T)/pr_sum # pr[a] is probability of taking action a

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

    def update_agent(self, s_new, meta_state_new, a, reward, learning_params, update_q_function=True):
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
        label : string
            Label returned by the MDP this step.
        learning_params : LearningParameters object
            Object storing parameters to be used in learning.
        """
        
        self.total_local_reward += reward

        if update_q_function == True:
            self.update_q_function(self.s, s_new, self.meta_state, meta_state_new, a, reward, learning_params)

        # Moving to the next state
        self.s = s_new
        self.meta_state = meta_state_new

    def update_q_function(self, s, s_new, meta_state, meta_state_new, a, reward, learning_params):
        """
        Update the q function using the action, states, and reward value.

        Parameters
        ----------
        s : int
            Index of the agent's previous state
        s_new : int
            Index of the agent's updated state
        a : int
            Action the agent took from state s
        reward : float
            Reward the agent achieved during this step
        learning_params : LearningParameters object
            Object storing parameters to be used in learning.
        """
        alpha = learning_params.alpha
        gamma = learning_params.gamma

        # Bellman update
        self.q[s, meta_state, a] = (1-alpha)*self.q[s, meta_state, a] + alpha*(reward + gamma*np.amax(self.q[s_new, meta_state_new]))