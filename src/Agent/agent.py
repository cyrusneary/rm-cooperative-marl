from reward_machines.sparse_reward_machine import SparseRewardMachine
from tester.tester import Tester
import numpy as np
import random, time, os
import matplotlib.pyplot as plt

class Agent:
    """
    Class meant to represent an individual RM-based learning agent.
    The agent maintains a representation of its own q-function and accumulated reward
    which are updated across training episodes.
    The agent also has a representation of its own local reward machine, which it uses
    for learning, and of its state in the world/reward machine.
    
    Note: Users of this class must manually reset the world state and the reward machine
    state when starting a new episode by calling self.initialize_world() and 
    self.initialize_reward_machine().
    """
    def __init__(self, rm_file, s_i, num_states, actions, agent_id):
        """
        Initialize agent object.

        Parameters
        ----------
        rm_file : str
            File path pointing to the reward machine this agent is meant to use for learning.
        s_i : int
            Index of initial state.
        actions : list
            List of actions available to the agent.
        agent_id : int
            Index of this agent.
        """
        self.rm_file = rm_file
        self.agent_id = agent_id
        self.s_i = s_i
        self.s = s_i
        self.actions = actions
        self.num_states = num_states

        self.rm = SparseRewardMachine(self.rm_file)
        self.u = self.rm.get_initial_state()
        self.local_event_set = self.rm.get_events()
        
        self.q = np.zeros([num_states, len(self.rm.U), len(self.actions)])
        self.total_local_reward = 0
        self.is_task_complete = 0

    def reset_state(self):
        """
        Reset the agent to the initial state of the environment.
        """
        self.s = self.s_i

    def initialize_reward_machine(self):
        """
        Reset the state of the reward machine to the initial state and reset task status.
        """
        self.u = self.rm.get_initial_state()
        self.is_task_complete = 0

    def is_local_event_available(self, label):
        if label: # Only try accessing the first event in label if it exists
            event = label[0]
            return self.rm.is_event_available(self.u, event)
        else:
            return False

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
            pr_sum = np.sum(np.exp(self.q[self.s, self.u, :] * T))
            pr = np.exp(self.q[self.s, self.u, :] * T)/pr_sum # pr[a] is probability of taking action a

            # If any q-values are so large that the softmax function returns infinity, 
            # make the corresponding actions equally likely
            if any(np.isnan(pr)):
                print('BOLTZMANN CONSTANT TOO LARGE IN ACTION-SELECTION SOFTMAX.')
                temp = np.array(np.isnan(pr), dtype=float)
                pr = temp / np.sum(temp)

            pr_select = np.zeros(len(self.actions) + 1)
            pr_select[0] = 0
            for i in range(len(self.actions)):
                pr_select[i+1] = pr_select[i] + pr[i]

            randn = random.random()
            for a in self.actions:
                if randn >= pr_select[a] and randn <= pr_select[a+1]:
                    a_selected = a
                    break

            # best_actions = np.where(self.q[self.s, self.u, :] == np.max(self.q[self.s, self.u, :]))[0]
            # a_selected = random.choice(best_actions)
        
        a = a_selected

        return self.s, a

    def update_agent(self, s_new, a, reward, label, learning_params, update_q_function=True):
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

        # Keep track of the RM location at the start of the 
        u_start = self.u

        for event in label: # there really should only be one event in the label provided to an individual agent
            # Update the agent's RM
            u2 = self.rm.get_next_state(self.u, event)
            self.u = u2
        
        self.total_local_reward += reward

        if update_q_function == True:
            self.update_q_function(self.s, s_new, u_start, self.u, a, reward, learning_params)

        # Moving to the next state
        self.s = s_new

        if self.rm.is_terminal_state(self.u):
            # Completed task. Set flag.
            self.is_task_complete = 1

    def update_q_function(self, s, s_new, u, u_new, a, reward, learning_params):
        """
        Update the q function using the action, states, and reward value.

        Parameters
        ----------
        s : int
            Index of the agent's previous state
        s_new : int
            Index of the agent's updated state
        u : int
            Index of the agent's previous RM state
        U_new : int
            Index of the agent's updated RM state
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
        self.q[s][u][a] = (1-alpha)*self.q[s][u][a] + alpha*(reward + gamma*np.amax(self.q[s_new][u_new]))