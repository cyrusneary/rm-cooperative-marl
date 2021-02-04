from tester.tester import Tester
from tester.tester_params import TestingParameters
from tester.learning_params import LearningParameters
import os

def rendezvous_config(num_times, num_agents):
    """
    Function setting the experiment parameters and environment.

    Output
    ------
    Tester : tester object
        Object containing the information necessary to run this experiment.
    """
    base_file_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

    joint_rm_file = os.path.join(base_file_path, 'experiments', 'gridworld_many_agent_rendezvous', '{}_agent_rendezvous_rm.txt'.format(num_agents))

    local_rm_files = []
    for i in range(num_agents):
        local_rm_string = os.path.join(base_file_path, 'experiments', 'gridworld_many_agent_rendezvous', 'coordination_experiment_agent{}.txt'.format(i+1))
        local_rm_files.append(local_rm_string)

    step_unit = 1000

    # configuration of testing params
    testing_params = TestingParameters()
    testing_params.test = True
    testing_params.test_freq =  1*step_unit
    testing_params.num_steps = step_unit

    # configuration of learning params
    learning_params = LearningParameters()
    learning_params.gamma = 0.9
    learning_params.alpha = 0.8
    learning_params.T = 50
    learning_params.initial_epsilon = 0.0 # Set epsilon to zero to turn off epsilon-greedy exploration (only using boltzmann)
    learning_params.tabular_case = True
    learning_params.max_timesteps_per_task = step_unit
    learning_params.relearn_period = 30
    learning_params.enter_loop = 10

    tester = Tester(learning_params, testing_params)
    tester.total_steps = 150 * step_unit
    tester.min_steps = 1

    tester.rm_test_file = joint_rm_file
    tester.rm_learning_file_list = local_rm_files

    tester.num_times = num_times
    tester.num_agents = num_agents

    # Set the environment settings for the experiment
    env_settings = dict()
    env_settings['Nr'] = 10
    env_settings['Nc'] = 10
    env_settings['initial_states'] = [0, 3, 20, 8, 90, 40, 70, 49, 96, 69]
    env_settings['rendezvous_loc'] = (3,4)
    env_settings['goal_locations'] = [(9,7), (7,9), (2,9), (9,9), (0,9), (7,0), (4,0), (5,0), (6,9), (8,0)]
    env_settings['p'] = 0.98

    tester.env_settings = env_settings
    tester.experiment = 'rendezvous'

    return tester