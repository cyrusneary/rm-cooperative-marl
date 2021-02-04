import numpy as np
import random, time

from tester.tester import Tester
from Agent.ihrl_agent import IhrlAgent
from Environments.rendezvous.gridworld_env import GridWorldEnv
from Environments.rendezvous.multi_agent_gridworld_env import MultiAgentGridWorldEnv
from Environments.coop_buttons.multi_agent_buttons_env import MultiAgentButtonsEnv
import matplotlib.pyplot as plt
import math

def run_ihrl_training(epsilon,
                    tester,
                    agent_list,
                    show_print=True):
    """
    This code runs one i-hrl training episode. q-functions, and accumulated reward values of agents
    are updated accordingly. If the appropriate number of steps have elapsed, this function will
    additionally run a test episode.

    Parameters
    ----------
    epsilon : float
        Numerical value in (0,1) representing likelihood of choosing a random action.
    tester : Tester object
        Object containing necessary information for current experiment.
    agent_list : list of Agent objects
        Agent objects to be trained and tested.
    show_print : bool
        Optional flag indicating whether or not to print output statements to terminal.
    """
    # Initializing parameters and the game
    learning_params = tester.learning_params
    testing_params = tester.testing_params
    
    num_agents = len(agent_list)
    if tester.experiment == 'rendezvous':
        training_env = MultiAgentGridWorldEnv(tester.rm_test_file, num_agents, tester.env_settings)
    if tester.experiment == 'buttons':
        training_env = MultiAgentButtonsEnv(tester.rm_test_file, num_agents, tester.env_settings)

    for i in range(num_agents):
        agent_list[i].reset_state()
        agent_list[i].reset_option()

    s_team = np.full(num_agents, -1, dtype=int)
    a_team = np.full(num_agents, -1, dtype=int)
    testing_reward = 0

    mc_rewards = dict()
    for i in range(num_agents):
        mc_rewards[i] = []

    num_steps = learning_params.max_timesteps_per_task

    for t in range(num_steps):
        # Update step count
        tester.add_step()

        for i in range(num_agents):
            if t == 0:
                current_meta_state = training_env.get_meta_state(i)
                avail_meta_action_indeces = training_env.get_avail_meta_action_indeces(i)
                meta_action = agent_list[i].get_next_meta_action(current_meta_state, avail_meta_action_indeces, epsilon, learning_params)
                agent_list[i].current_option = agent_list[i].options_list[meta_action]
                agent_list[i].option_start_state = training_env.get_meta_state(i)
                agent_list[i].option_complete = False

            if agent_list[i].option_complete:
                # Update the meta controller
                option_start_state = agent_list[i].option_start_state
                current_meta_state = training_env.get_meta_state(i)
                meta_action = agent_list[i].options_list.index(agent_list[i].current_option)
                agent_list[i].update_meta_q_function(option_start_state, current_meta_state, meta_action, mc_rewards[i], learning_params)

                # choose the next meta action
                avail_meta_action_indeces = training_env.get_avail_meta_action_indeces(i)
                meta_action = agent_list[i].get_next_meta_action(current_meta_state, avail_meta_action_indeces, epsilon, learning_params)
                agent_list[i].current_option = agent_list[i].options_list[meta_action]
                agent_list[i].option_start_state = current_meta_state
                agent_list[i].option_complete = False
                mc_rewards[i] = []

        # Perform a team step
        for i in range(num_agents):
            s = agent_list[i].s
            s_team[i] = s
            a_team[i] = agent_list[i].get_next_action(epsilon, learning_params)

        r, _, s_team_next = training_env.environment_step(s_team, a_team)

        for i in range(num_agents):
            mc_rewards[i].append(r)

        completed_options = training_env.get_completed_options(s_team_next)

        for i in range(num_agents):
            # a = training_env.get_last_action(i)
            avail_options = training_env.get_avail_options(i)
            agent_list[i].update_agent(s_team_next[i], avail_options, a_team[i], r, completed_options, learning_params, update_q_function=True)
            if agent_list[i].current_option in completed_options:
                agent_list[i].option_complete = True

        # If enough steps have elapsed, test and save the performance of the agents.
        if testing_params.test and tester.get_current_step() % testing_params.test_freq == 0:
            t_init = time.time()
            step = tester.get_current_step()

            agent_list_copy = []

            # Need to create a copy of the agent for testing. If we pass the agent directly
            # mid-episode to the test function, the test will reset the world-state and reward machine 
            # state before the training episode has been completed.
            for i in range(num_agents):
                options_list = agent_list[i].options_list
                s_i = agent_list[i].s_i
                num_states = agent_list[i].num_states
                num_meta_states = agent_list[i].num_meta_states
                actions = agent_list[i].actions
                agent_id = agent_list[i].agent_id
                agent_copy = IhrlAgent(options_list, s_i, num_states, num_meta_states, actions, agent_id)
                # Pass only the q-functions by reference so that the testing updates the original agent's q-function.
                agent_copy.q_dict = agent_list[i].q_dict
                agent_copy.meta_q = agent_list[i].meta_q

                agent_list_copy.append(agent_copy)

            # Run a test of the performance of the agents
            testing_reward, trajectory, testing_steps = run_ihrl_test(agent_list_copy,
                                                                        tester,
                                                                        learning_params,
                                                                        testing_params,
                                                                        show_print=show_print)
            
            if 0 not in tester.results.keys():
                tester.results[0] = {}
            if step not in tester.results[0]:
                tester.results[0][step] = []
            tester.results[0][step].append(testing_reward)

            # Save the testing trace
            if 'trajectories' not in tester.results.keys():
                tester.results['trajectories'] = {}
            if step not in tester.results['trajectories']:
                tester.results['trajectories'][step] = []
            tester.results['trajectories'][step].append(trajectory)

            # Save how many steps it took to complete the task
            if 'testing_steps' not in tester.results.keys():
                tester.results['testing_steps'] = {}
            if step not in tester.results['testing_steps']:
                tester.results['testing_steps'][step] = []
            tester.results['testing_steps'][step].append(testing_steps)

            if len(tester.steps) == 0 or tester.steps[-1] < step:
                tester.steps.append(step)

        # If the task is complete, update the meta controllers and stop trying to complete it.
        env_rm_state = training_env.u
        if training_env.reward_machine.is_terminal_state(env_rm_state):
            for i in range(num_agents):
                # Update the meta controller
                option_start_state = agent_list[i].option_start_state
                current_meta_state = training_env.get_meta_state(i)
                meta_action = agent_list[i].options_list.index(agent_list[i].current_option)
                agent_list[i].update_meta_q_function(option_start_state, current_meta_state, meta_action, mc_rewards[i], learning_params)
            
            # Make sure we've run at least the minimum number of training steps before breaking the loop
            if tester.stop_task(t):
                break

        # checking the steps time-out
        if tester.stop_learning():
            break

def run_ihrl_test(agent_list,
                tester,
                learning_params,
                testing_params,
                show_print=True):
    """
    Run a test of the hrl method with the current q-function. 

    Parameters
    ----------
    agent_list : list of Agent objects
        Agent objects to be trained and tested.
    tester : Tester object
        Object containing necessary information for current experiment.
    learning_params : LearningParameters object
        Object storing parameters to be used in learning.
    Testing_params : TestingParameters object
        Object storing parameters to be used in testing.
    show_print : bool
        Optional flag indicating whether or not to print output statements to terminal.

    Ouputs
    ------
    testing_reard : float
        Reward achieved by agent during this test episode.
    trajectory : list
        List of dictionaries containing information on current step of test.
    step : int
        Number of testing steps required to complete the task.
    """
    num_agents = len(agent_list)
    if tester.experiment == 'rendezvous':
        testing_env = MultiAgentGridWorldEnv(tester.rm_test_file, num_agents, tester.env_settings)
    if tester.experiment == 'buttons':
        testing_env = MultiAgentButtonsEnv(tester.rm_test_file, num_agents, tester.env_settings)

    for i in range(num_agents):
        agent_list[i].reset_state()
        agent_list[i].reset_option()

    s_team = np.full(num_agents, -1, dtype=int)
    a_team = np.full(num_agents, -1, dtype=int)
    testing_reward = 0

    mc_rewards = dict()
    for i in range(num_agents):
        mc_rewards[i] = []

    trajectory = []
    step = 0

    # agent_list[0].meta_q[:,0] = 1
    # agent_list[0].meta_q[0,0] = 0
    # agent_list[0].meta_q[0,1] = 1
    # agent_list[0].meta_q[7,0] = 0
    # agent_list[0].meta_q[7,2] = 1

    # agent_list[1].meta_q[:,0] = 1
    # agent_list[1].meta_q[1,0] = 0
    # agent_list[1].meta_q[1,1] = 1
    # agent_list[1].meta_q[3,0] = 0
    # agent_list[1].meta_q[3,2] = 1

    # agent_list[2].meta_q[:,0] = 1
    # agent_list[2].meta_q[3,0] = 0
    # agent_list[2].meta_q[3,1] = 1

    # Starting interaction with the environment
    for t in range(testing_params.num_steps):
        step = step + 1

        for i in range(num_agents):
            if t == 0:
                current_meta_state = testing_env.get_meta_state(i)
                avail_meta_action_indeces = testing_env.get_avail_meta_action_indeces(i)
                meta_action = agent_list[i].get_next_meta_action(current_meta_state, avail_meta_action_indeces, -1, learning_params)
                agent_list[i].current_option = agent_list[i].options_list[meta_action]
                agent_list[i].option_start_state = testing_env.get_meta_state(i)
                agent_list[i].option_complete = False

            if agent_list[i].option_complete:
                # choose the next meta action
                current_meta_state = testing_env.get_meta_state(i)
                avail_meta_action_indeces = testing_env.get_avail_meta_action_indeces(i)
                meta_action = agent_list[i].get_next_meta_action(current_meta_state, avail_meta_action_indeces, -1, learning_params)
                agent_list[i].current_option = agent_list[i].options_list[meta_action]
                agent_list[i].option_start_state = current_meta_state
                agent_list[i].option_complete = False
                mc_rewards[i] = []

        # print('Meta sate: {}, Agent 1: {}, Agent 2: {}, Agent 3: {}'.format(testing_env.get_meta_state(1), agent_list[0].current_option, agent_list[1].current_option, agent_list[2].current_option))

        # Perform a team step
        for i in range(num_agents):
            s = agent_list[i].s
            s_team[i] = s
            a_team[i] = agent_list[i].get_next_action(-1.0, learning_params)

        # trajectory.append({'s' : np.array(s_team, dtype=int), 'a' : np.array(a_team, dtype=int), 'meta_state': testing_env.get_meta_state(i)})

        r, _, s_team_next = testing_env.environment_step(s_team, a_team)
        testing_reward = testing_reward + r

        for i in range(num_agents):
            mc_rewards[i].append(r)

        completed_options = testing_env.get_completed_options(s_team_next)

        for i in range(num_agents):
            # a = testing_env.get_last_action(i)
            avail_options = testing_env.get_avail_options(i)
            agent_list[i].update_agent(s_team_next[i], avail_options, a_team[i], r, completed_options, learning_params, update_q_function=False)
            if agent_list[i].current_option in completed_options:
                agent_list[i].option_complete = True

        # If the task is complete, update all meta controllers and stop trying to complete it.
        env_rm_state = testing_env.u
        if testing_env.reward_machine.is_terminal_state(env_rm_state):
            break

    if show_print:
        print('Reward of {} achieved in {} steps. Current step: {} of {}'.format(testing_reward, step, tester.current_step, tester.total_steps))

    return testing_reward, trajectory, step

def run_ihrl_experiment(tester,
                            num_agents,
                            num_times,
                            show_print=True):
    """
    Run the entire q-learning with reward machines experiment a number of times specified by num_times.

    Inputs
    ------
    tester : Tester object
        Test object holding true reward machine and all information relating
        to the particular tasks, world, learning parameters, and experimental results.
    num_agents : int
        Number of agents in this experiment.
    num_times : int
        Number of times to run the entire experiment (restarting training from scratch).
    show_print : bool
        Flag indicating whether or not to output text to the terminal.
    """
    
    learning_params = tester.learning_params

    for t in range(num_times):
        # Reseting default step values
        tester.restart()

        rm_test_file = tester.rm_test_file
        rm_learning_file_list = tester.rm_learning_file_list

        # Verify that the number of local reward machines matches the number of agents in the experiment.
        assertion_string = "Number of specified local reward machines must match specified number of agents."
        assert (len(rm_learning_file_list) == num_agents), assertion_string

        if tester.experiment == 'rendezvous':
            testing_env = MultiAgentGridWorldEnv(rm_test_file, num_agents, tester.env_settings)
        if tester.experiment == 'buttons':
            testing_env = MultiAgentButtonsEnv(rm_test_file, num_agents, tester.env_settings)

        num_states = testing_env.num_states

        # Create the a list of agents for this experiment
        agent_list = [] 
        for i in range(num_agents):
            # The actions available to individual agents should be specified by the TRUE multi-agent environment.
            actions = testing_env.get_actions(i)
            s_i = testing_env.get_initial_state(i)
            num_meta_states = testing_env.get_num_meta_states(i)
            options_list = testing_env.get_options_list(i)
            agent_list.append(IhrlAgent(options_list, s_i, num_states, num_meta_states, actions, i))

        num_episodes = 0

        # Task loop
        epsilon = learning_params.initial_epsilon

        while not tester.stop_learning():
            num_episodes += 1

            epsilon = epsilon * 0.99

            run_ihrl_training(epsilon,
                                tester,
                                agent_list,
                                show_print=show_print)

        # Backing up the results
        print('Finished iteration ',t)

    tester.agent_list = agent_list

    plot_multi_agent_results(tester, num_agents)

def plot_multi_agent_results(tester, num_agents):
    """
    Plot the results stored in tester.results for each of the agents.
    """

    prc_25 = list()
    prc_50 = list()
    prc_75 = list()

    # Buffers for plots
    current_step = list()
    current_25 = list()
    current_50 = list()
    current_75 = list()
    steps = list()

    plot_dict = tester.results['testing_steps']

    for step in plot_dict.keys():
        if len(current_step) < 10:
            current_25.append(np.percentile(np.array(plot_dict[step]),25))
            current_50.append(np.percentile(np.array(plot_dict[step]),50))
            current_75.append(np.percentile(np.array(plot_dict[step]),75))
            current_step.append(sum(plot_dict[step])/len(plot_dict[step]))
        else:
            current_step.pop(0)
            current_25.pop(0)
            current_50.pop(0)
            current_75.pop(0)
            current_25.append(np.percentile(np.array(plot_dict[step]),25))
            current_50.append(np.percentile(np.array(plot_dict[step]),50))
            current_75.append(np.percentile(np.array(plot_dict[step]),75))
            current_step.append(sum(plot_dict[step])/len(plot_dict[step]))

        prc_25.append(sum(current_25)/len(current_25))
        prc_50.append(sum(current_50)/len(current_50))
        prc_75.append(sum(current_75)/len(current_75))
        steps.append(step)

    plt.plot(steps, prc_25, alpha=0)
    plt.plot(steps, prc_50, color='red')
    plt.plot(steps, prc_75, alpha=0)
    plt.grid()
    plt.fill_between(steps, prc_50, prc_25, color='red', alpha=0.25)
    plt.fill_between(steps, prc_50, prc_75, color='red', alpha=0.25)
    plt.ylabel('Testing Steps to Task Completion', fontsize=15)
    plt.xlabel('Training Steps', fontsize=15)
    plt.locator_params(axis='x', nbins=5)
    plt.show()