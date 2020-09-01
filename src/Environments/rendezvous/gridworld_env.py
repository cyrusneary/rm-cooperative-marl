import random, math, os
import numpy as np
from enum import Enum

import sys
sys.path.append('../')
sys.path.append('../../')
from reward_machines.sparse_reward_machine import SparseRewardMachine

"""
Enum with the actions that the agent can execute
"""
class Actions(Enum):
    up    = 0 # move up
    right = 1 # move right
    down  = 2 # move down
    left  = 3 # move left
    none  = 4 # none 

class GridWorldEnv:

    def __init__(self, rm_file, agent_id, env_settings):
        """
        Initialize gridworld environment.

        Parameters
        ----------
        rm_file : string
            File path leading to the text file containing the reward machine
            encoding this environment's reward function.
        agent_id : int
            Index {0,1} indicating which agent
        env_settings : dict
            Dictionary of environment settings
        """
        self.env_settings = env_settings
        self.agent_id = agent_id
        self._load_map()
        self.reward_machine = SparseRewardMachine(rm_file)

        self.u = self.reward_machine.get_initial_state()
        self.last_action = -1 # Initialize last action to garbage value

    def _load_map(self):
        """
        Initialize the environment.
        """
        self.Nr = self.env_settings['Nr']
        self.Nc = self.env_settings['Nc']

        initial_states = self.env_settings['initial_states']

        self.s_i = initial_states[self.agent_id-1]
        self.objects = {}
        self.objects[self.env_settings['rendezvous_loc']] = "w" # rendezvous location
        goal_locations = self.env_settings['goal_locations']
        self.objects[goal_locations[self.agent_id-1]] = "g" # goal location

        self.p = self.env_settings['p']
        
        self.num_states = self.Nr * self.Nc

        self.actions = [Actions.up.value, Actions.right.value, Actions.left.value, Actions.down.value, Actions.none.value]
        
        # Define forbidden transitions corresponding to map edges
        self.forbidden_transitions = set()
        
        for row in range(self.Nr):
            self.forbidden_transitions.add((row, 0, Actions.left)) # If in left-most column, can't move left.
            self.forbidden_transitions.add((row, self.Nc - 1, Actions.right)) # If in right-most column, can't move right.
        for col in range(self.Nc):
            self.forbidden_transitions.add((0, col, Actions.up)) # If in top row, can't move up
            self.forbidden_transitions.add((self.Nr - 1, col, Actions.down)) # If in bottom row, can't move down

    def environment_step(self, s, a):
        """
        Execute action a from state s.

        Parameters
        ----------
        s : int
            Index representing the current environment state.
        a : int
            Index representing the action being taken.

        Outputs
        -------
        r : float
            Reward achieved by taking action a from state s.
        l : list
            List of events occuring at this step.
        s_next : int
            Index of next state.
        """
        s_next, last_action = self.get_next_state(s,a)
        self.last_action = last_action

        l = self.get_mdp_label(s, s_next, self.u)
        r = 0

        for e in l:
            # Get the new reward machine state and the reward of this step
            u2 = self.reward_machine.get_next_state(self.u, e)
            r = r + self.reward_machine.get_reward(self.u, u2)
            # Update the reward machine state
            self.u = u2

        return r, l, s_next

    def get_mdp_label(self, s, s_next, u):
        """
        """
        row, col = self.get_state_description(s)
        row_next, col_next = self.get_state_description(s_next)

        l = []

        thresh = 0.3
        
        if u == 0 and (row_next, col_next) in self.objects:
                if self.objects[(row_next, col_next)] == 'w':
                    l.append('r{}'.format(self.agent_id))
        elif u == 1:
            if not((row_next, col_next) in self.objects):
                l.append('l{}'.format(self.agent_id))
            elif self.objects[(row_next,col_next)] == 'w' and (row,col) in self.objects:
                if self.objects[(row,col)] == 'w' and np.random.random() <= thresh:
                    l.append('r')
        elif u == 2:
            if (row_next, col_next) in self.objects:
                if self.objects[(row_next,col_next)] == 'g':
                    l.append('g{}'.format(self.agent_id))
        return l

    def get_next_state(self, s, a):
        """
        Get the next state in the environment given action a is taken from state s.
        Update the last action that was truly taken due to MDP slip.

        Parameters
        ----------
        s : int
            Index of the current state.
        a : int
            Action to be taken from state s.

        Outputs
        -------
        s_next : int
            Index of the next state.
        last_action :int
            Last action taken by agent due to slip proability.
        """
        slip_p = [self.p, (1-self.p)/2, (1-self.p)/2]
        check = random.random()

        row, col = self.get_state_description(s)

        # up    = 0
        # right = 1 
        # down  = 2 
        # left  = 3 

        if (check<=slip_p[0]) or (a == Actions.none.value):
            a_ = a

        elif (check>slip_p[0]) & (check<=(slip_p[0]+slip_p[1])):
            if a == 0: 
                a_ = 3
            elif a == 2: 
                a_ = 1
            elif a == 3: 
                a_ = 2
            elif a == 1: 
                a_ = 0

        else:
            if a == 0: 
                a_ = 1
            elif a == 2: 
                a_ = 3
            elif a == 3: 
                a_ = 0
            elif a == 1: 
                a_ = 2

        action_ = Actions(a_)
        if (row, col, action_) not in self.forbidden_transitions:
            if action_ == Actions.up:
                row -= 1
            if action_ == Actions.down:
                row += 1
            if action_ == Actions.left:
                col -= 1
            if action_ == Actions.right:
                col += 1

        s_next = self.get_state_from_description(row, col)

        last_action = a_
        return s_next, last_action

    def get_state_from_description(self, row, col):
        """
        Given a (row, column) index description of gridworld location, return
        index of corresponding state.

        Parameters
        ----------
        row : int
            Index corresponding to the row location of the state in the gridworld.
        col : int
            Index corresponding to the column location of the state in the gridworld.
        
        Outputs
        -------
        s : int
            The index of the gridworld state corresponding to location (row, col).
        """
        return self.Nc * row + col

    def get_state_description(self, s):
        """
        Return the row and column indeces of state s in the gridworld.

        Parameters
        ----------
        s : int
            Index of the gridworld state.

        Outputs
        -------
        row : int
            The row index of state s in the gridworld.
        col : int
            The column index of state s in the gridworld.
        """
        row = np.floor_divide(s, self.Nr)
        col = np.mod(s, self.Nc)

        return (row, col)

    def get_actions(self):
        """
        Returns the list with the actions that the agent can perform
        """
        return self.actions

    def get_last_action(self):
        """
        Returns agent's last action
        """
        return self.last_action

    def get_initial_state(self):
        """
        Outputs
        -------
        s_i : int
            Index of agent's initial state.
        """
        return self.s_i

    def show(self, s):
        """
        Create a visual representation of the current state of the gridworld.

        Parameters
        ----------
        s : int
            Index of the current state
        """
        display = np.zeros((self.Nr, self.Nc))
        
        # Display the location of key points in world
        for loc in self.objects.keys():
            display[loc] = 1

        # Display the location of the agent in the world
        row, col = self.get_state_description(s)
        display[row,col] = np.nan

        print(display)

def play():
    agent_id = 1

    # Set the environment settings for the experiment
    env_settings = dict()
    env_settings['Nr'] = 10
    env_settings['Nc'] = 10
    env_settings['initial_states'] = [0, 3, 20, 8, 90, 40, 70, 49, 96, 69]
    env_settings['rendezvous_loc'] = (3,4)
    env_settings['goal_locations'] = [(9,7), (7,9), (2,9), (9,9), (0,9), (7,0), (4,0), (5,0), (6,9), (8,0)]
    env_settings['p'] = 1.0

    base_file_dir = os.path.abspath(os.path.join(os.getcwd(), '../../..'))
    rm_string = os.path.join(base_file_dir, 'experiments', 'gridworld_many_agent_rendezvous', 'coordination_experiment_agent{}.txt'.format(agent_id))
    game = GridWorldEnv(rm_string, agent_id, env_settings)

    # User inputs
    str_to_action = {"w":Actions.up.value,"d":Actions.right.value,"s":Actions.down.value,"a":Actions.left.value,"x":Actions.none.value}

    s = game.get_initial_state()

    while True:
        # Showing game
        game.show(s)

        # Getting action
        print("\nAction? ", end="")
        a = input()
        print()
        # Executing action
        if a in str_to_action:
            r, l, s = game.environment_step(s, str_to_action[a])
        
            print("---------------------")
            print("Next States: ", s)
            print("Label: ", l)
            print("Reward: ", r)
            print("RM state: ", game.u)
            print("---------------------")

            if game.reward_machine.is_terminal_state(game.u): # Game Over
                    break 
            
        else:
            print("Forbidden action")
    game.show(s)
    
# This code allow to play a game (for debugging purposes)
if __name__ == '__main__':
    play()