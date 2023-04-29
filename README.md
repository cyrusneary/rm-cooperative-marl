# DQPRM - Decentralized Q-Learning with Projected Reward Machines

This project (Reward Machines for Cooperative Multi-Agent Reinforcement Learning: https://arxiv.org/abs/2007.01962) studies reward machines for multi-agent q-learning of temporally extended cooperative tasks.

## Installation instructions

DQPRM requires Python 3.6 with libraries numpy and matplotlib.

## Running examples

To configure the example to run, open run.py and set:

experiment = 'rendezvous' - DQPRM rendezvous experiment

experiment = 'centralized_rendezvous' - CQRM rendezvous experiment

experiment = 'ihrl_rendezvous' - I-hL rendezvous experiment

experiment = 'iql_rendezvous' - IQL rendezvous experiment

experiment = 'buttons' - DQPRM buttons experiment

experiment = 'ihrl_buttons' - I-hL buttons experiment

experiment = 'iql_buttons' - IQL buttons experiment

If running one of the rendezvous experiments, set num_agents to a number in [2,10] to control the number of agents involved in the experiment.

Use rendezvous_config.py and buttons_config.py to set the learning parameters and the parameters of rendezvous and buttons environments respectively.

To run the example, run src>>python run.py

## Acknowledgments

Several files in our implementation adapt code originally included in https://bitbucket.org/RToroIcarte/qrm. We thank the authors of this work, who study the use of reward machines for q-learning in the single-agent setting.
