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

class MultiAgentGridWorldEnv:

    def __init__(self, rm_file, num_agents, env_settings):
        """
        Initialize gridworld environment.

        Parameters
        ----------
        rm_file : string
            File path leading to the text file containing the reward machine
            encoding this environment's reward function.
        num_agents : int
            Number of agents in the environment.
        env_settings : dict
            Dictionary of environment settings
        """
        self.env_settings = env_settings
        self.num_agents = num_agents
        
        self._load_map()
        self.reward_machine = SparseRewardMachine(rm_file)

        self.u = self.reward_machine.get_initial_state()
        self.last_action = np.full(self.num_agents, -1, dtype=int) #Initialize last action with garbage values

        self.rendezvous_complete = False
        self.reached_goal_flag = []
        for i in range(num_agents):
            self.reached_goal_flag.append(False)

    def _load_map(self):
        """
        Initialize the environment.
        """
        self.Nr = self.env_settings['Nr']
        self.Nc = self.env_settings['Nc']

        initial_states = self.env_settings['initial_states']

        # Define Initial states of all agents
        self.s_i = np.full(self.num_agents, -1, dtype=int)
        for i in range(self.num_agents):
            self.s_i[i] = initial_states[i]

        # Populate the map with markers
        self.objects = {}
        self.objects[self.env_settings['rendezvous_loc']] = "w" # rendezvous location
        goal_locations = self.env_settings['goal_locations']
        for i in range(self.num_agents):
            self.objects[goal_locations[i]] = 'g{}'.format(i+1)

        self.p = self.env_settings['p']

        # Set the available actions of all agents. For now all agents have same action set.
        actions = np.array([Actions.up.value, Actions.right.value, Actions.left.value, Actions.down.value, Actions.none.value], dtype=int)
        self.actions = np.full((self.num_agents, len(actions)), -2, dtype=int)
        for i in range(self.num_agents):
            self.actions[i] = actions

        self.num_states = self.Nr * self.Nc
        
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
        Execute collective action a from collective state s. Return the resulting reward,
        mdp label, and next state. Update the last action taken by each agent.

        Parameters
        ----------
        s : numpy integer array
            Array of integers representing the environment states of the various agents.
            s[id] represents the state of the agent indexed by index "id".
        a : numpy integer array
            Array of integers representing the actions selected by the various agents.
            a[id] represents the desired action to be taken by the agent indexed by "id.

        Outputs
        -------
        r : float
            Reward achieved by taking action a from state s.
        l : string
            MDP label emitted this step.
        s_next : numpy integer array
            Array of indeces of next team state.
        """
        s_next = np.full(self.num_agents, -1, dtype=int)

        for i in range(self.num_agents):
            s_next[i], last_action = self.get_next_state(s[i], a[i])
            self.last_action[i] = last_action

        rendezvous_row = self.env_settings['rendezvous_loc'][0]
        rendezvous_col = self.env_settings['rendezvous_loc'][1]

        rendezvous_state = self.get_state_from_description(rendezvous_row, rendezvous_col)

        if all(s[i] == rendezvous_state for i in range(self.num_agents)):
            self.rendezvous_complete = True

        l = self.get_mdp_label(s, s_next, self.u)
        r = 0

        for e in l:
            # Get the new reward machine state and the reward of this step
            u2 = self.reward_machine.get_next_state(self.u, e)
            r = r + self.reward_machine.get_reward(self.u, u2)
            # Update the reward machine state
            self.u = u2

        # If the rendezvous is complete, the agents can start reaching their goals 
        # (The reached_goal_flag list is used to update the meta-state)
        if self.rendezvous_complete:
            for i in range(self.num_agents):
                if 'g{}'.format(i+1) in l:
                    self.reached_goal_flag[i] = True

        return r, l, s_next

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
        last_action : int
            Last action the agent truly took because of slip probability.
        """
        slip_p = [self.p, (1-self.p)/2, (1-self.p)/2]
        check = random.random()

        row, col = self.get_state_description(s)

        # up    = 0
        # right = 1 
        # down  = 2 
        # left  = 3 
        # stay  = 4

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

    def get_actions(self, id):
        """
        Returns the list with the actions that a particular agent can perform.

        Parameters
        ----------
        id : int
            Index of the agent whose initial state is being queried.
        """
        return np.copy(self.actions[id])

    def get_last_action(self, id):
        """
        Returns a particular agent's last action.

        Parameters
        ----------
        id : int
            Index of the agent whose initial state is being queried.
        """
        return self.last_action[id]

    def get_team_action_array(self):
        """
        Returns the available actions of the entire team.

        Outputs
        -------
        actions : (num_agents x num_actions) numpy integer array
        """
        return np.copy(self.actions)
    
    def get_initial_state(self, id):
        """
        Returns the initial state of a particular agent.

        Parameters
        ----------
        id : int
            Index of the agent whose initial state is being queried.
        """
        return self.s_i[id]

    def get_initial_team_state(self):
        """
        Return the intial state of the collective multi-agent team.

        Outputs
        -------
        s_i : numpy integer array
            Array of initial state indices for the agents in the experiment.
        """
        return np.copy(self.s_i)

    ############## DQPRM-RELATED METHODS ########################################
    def get_mdp_label(self, s, s_next, u):
        """
        Get the mdp label resulting from transitioning from state s to state s_next.

        Parameters
        ----------
        s : numpy integer array
            Array of integers representing the environment states of the various agents.
            s[id] represents the state of the agent indexed by index "id".
        s_next : numpy integer array
            Array of integers representing the next environment states of the various agents.
            s_next[id] represents the next state of the agent indexed by index "id".
        u : int
            Index of the reward machine state

        Outputs
        -------
        l : string
            MDP label resulting from the state transition.
        """

        agent_all_on_wait = True

        l = []

        for i in range(self.num_agents):
            row, col = self.get_state_description(s[i])
            if (row, col) in self.objects:
                if not(self.objects[(row, col)] == 'w'):
                    agent_all_on_wait = False
            else:
                agent_all_on_wait = False

            row_next, col_next = self.get_state_description(s_next[i])
            if (row_next, col_next) not in self.objects:
                l.append("l{}".format(i+1))
            elif (row_next, col_next) in self.objects:
                if self.objects[(row_next, col_next)] == "w":
                    l.append('r{}'.format(i+1))
                elif self.objects[(row_next, col_next)] == "g{}".format(i+1):
                    l.append('g{}'.format(i+1))

        if agent_all_on_wait:
            # If all agents are on wait space, only return r
            l = []
            l.append('r')

        return l

    ################## HRL-RELATED METHODS ######################################
    def get_options_list(self, agent_id):
        """
        Return a list of strings representing the possible options for each agent.

        Input
        -----
        agent_id : int
            The id of the agent whose option list is to be returned.
        
        Output
        ------
        options_list : list
            list of strings representing the options avaialble to the agent.
        """

        options_list = []
        options_list.append('r{}'.format(agent_id + 1)) # Navigate to rendezvous
        options_list.append('g{}'.format(agent_id + 1)) # Navigate to goal

        return options_list

    def get_avail_options(self, agent_id):
        """
        Given the current metastate, get the available options. Some options are unavailable if 
        they are not possible to complete at the current stage of the task. In such circumstances
        we don't want the agents to update the corresponding option q-functions.
        """
        avail_options = self.get_options_list(agent_id)

        return avail_options

    def get_avail_meta_action_indeces(self, agent_id):
        """
        Get a list of the indeces corresponding to the currently available meta-action/option
        """
        avail_options = self.get_avail_options(agent_id)
        all_options_list = self.get_options_list(agent_id)
        avail_meta_action_indeces = []
        for option in avail_options:
            avail_meta_action_indeces.append(all_options_list.index(option))
        return avail_meta_action_indeces

    def get_completed_options(self, s):
        """
        Get a list of strings corresponding to options that are deemed complete in the team state described by s.

        Parameters
        ----------
        s : numpy integer array
            Array of integers representing the environment states of the various agents.
            s[id] represents the state of the agent indexed by index "id".

        Outputs
        -------
        completed_options : list
            list of strings corresponding to the completed options.
        """
        completed_options = []

        for i in range(self.num_agents):
            row, col = self.get_state_description(s[i])
            if (row, col) in self.objects:
                if self.objects[(row, col)] == "w":
                    completed_options.append('r{}'.format(i+1))
                elif self.objects[(row, col)] == "g{}".format(i+1):
                    completed_options.append('g{}'.format(i+1))
        
        return completed_options

    def get_num_meta_states(self, agent_id):
        """
        Return the number of meta states for the agent specified by agent_id.
        """
        return int(3)
        # return len(self.reward_machine.U)

    def get_meta_state(self, agent_id):
        """
        Return the meta-state that the agent should use for it's meta controller.

        Input
        -----
        s_team : numpy array
            s_team[i] is the state of agent i.
        agent_id : int
            Index of agent whose meta-state is to be returned.

        Output
        ------
        meta_state : int
            Index of the meta-state.
        """
        if not (self.rendezvous_complete):
            return 0
        elif self.rendezvous_complete and not(self.reached_goal_flag[agent_id]):
            return 1
        elif self.rendezvous_complete and self.reached_goal_flag[agent_id]:
            return 2

        # Suprisingly, directly using the states of the reward machine as the meta states 
        # did not work well.
        # return self.u

    ######################### TROUBLESHOOTING METHODS ################################

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
            display[loc] = 9

        # Display the agents
        for i in range(self.num_agents):
            row, col = self.get_state_description(s[i])
            display[row, col] = i + 1

        print(display)

def play():
    n = 2 # num agents

    # Set the environment settings for the experiment
    env_settings = dict()
    env_settings['Nr'] = 10
    env_settings['Nc'] = 10
    env_settings['initial_states'] = [0, 3, 20, 8, 90, 40, 70, 49, 96, 69]
    env_settings['rendezvous_loc'] = (3,4)
    env_settings['goal_locations'] = [(9,7), (7,9), (2,9), (9,9), (0,9), (7,0), (4,0), (5,0), (6,9), (8,0)]
    env_settings['p'] = 1.0

    base_file_dir = os.path.abspath(os.path.join(os.getcwd(), '../../..'))
    rm_string = os.path.join(base_file_dir, 'experiments', 'gridworld_many_agent_rendezvous', '{}_agent_rendezvous_rm.txt'.format(n))

    game = MultiAgentGridWorldEnv(rm_string, n, env_settings)

    # User inputs
    str_to_action = {"w":Actions.up.value,"d":Actions.right.value,"s":Actions.down.value,"a":Actions.left.value,"x":Actions.none.value}

    s = game.get_initial_team_state()
    print(s)

    while True:
        # Showing game
        game.show(s)

        # Getting action
        a = np.full(n, -1, dtype=int)
        
        for i in range(n):
            print('\nAction{}?'.format(i+1), end='')
            usr_inp = input()
            print()

            if not(usr_inp in str_to_action):
                print('forbidden action')
                a[i] = str_to_action['x']
            else:
                print(str_to_action[usr_inp])
                a[i] = str_to_action[usr_inp]

        r, l, s = game.environment_step(s, a)
        
        print("---------------------")
        print("Next States: ", s)
        print("Label: ", l)
        print("Reward: ", r)
        print("RM state: ", game.u)
        print("meta state: ", game.get_meta_state(0))
        print("---------------------")

        if game.reward_machine.is_terminal_state(game.u): # Game Over
                break 
    game.show(s)
    
# This code allow to play a game (for debugging purposes)
if __name__ == '__main__':
    play()