
class LearningParameters:
    def __init__(self, lr=0.001, max_timesteps_per_task=100000, buffer_size=50000,
                print_freq=1, exploration_fraction=0.1, exploration_final_eps=0.02,
                train_freq=1, batch_size=32, 
                learning_starts=1000, gamma=0.99, target_network_update_freq=500,
                tabular_case = False, use_double_dqn = False, use_random_maps = False,
                prioritized_replay=False, prioritized_replay_alpha=0.6, prioritized_replay_beta0=0.4,
                prioritized_replay_beta_iters=None, prioritized_replay_eps=1e-6,
                num_hidden_layers=-1, num_neurons=-1):
        """Parameters
        -------
        lr: float
            learning rate for adam optimizer
        max_timesteps_per_task: int
            number of env steps to optimizer for per task
        buffer_size: int
            size of the replay buffer
        exploration_fraction: float
            fraction of entire training period over which the exploration rate is annealed
        exploration_final_eps: float
            final value of random action probability
        train_freq: int
            update the model every `train_freq` steps.
            set to None to disable printing
        batch_size: int
            size of a batched sampled from replay buffer for training
        print_freq: int
            how often to print out training progress
            set to None to disable printing
        learning_starts: int
            how many steps of the model to collect transitions for before learning starts
        gamma: float
            discount factor
        target_network_update_freq: int
            update the target network every `target_network_update_freq` steps.
        tabular_case: bool
            if True, we solve the problem without an state approx
            if False, we solve the problem using a neuralnet
        prioritized_replay: True
            if True prioritized replay buffer will be used.
        prioritized_replay_alpha: float
            alpha parameter for prioritized replay buffer
        prioritized_replay_beta0: float
            initial value of beta for prioritized replay buffer
        prioritized_replay_beta_iters: int
            number of iterations over which beta will be annealed from initial value
            to 1.0. If set to None equals to max_timesteps.
        prioritized_replay_eps: float
            epsilon to add to the TD errors when updating priorities.    
        """
        self.lr = lr
        self.max_timesteps_per_task = max_timesteps_per_task
        self.buffer_size = buffer_size
        self.exploration_fraction = exploration_fraction
        self.exploration_final_eps = exploration_final_eps
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.print_freq = print_freq
        self.learning_starts = learning_starts
        self.gamma = gamma
        self.target_network_update_freq = target_network_update_freq
        # attributes for the tabular case
        self.tabular_case = tabular_case
        self.use_double_dqn = use_double_dqn
        self.use_random_maps = use_random_maps
        # Prioritized experience replay
        self.prioritized_replay = prioritized_replay
        self.prioritized_replay_alpha = prioritized_replay_alpha
        self.prioritized_replay_beta0 = prioritized_replay_beta0
        self.prioritized_replay_beta_iters = prioritized_replay_beta_iters
        self.prioritized_replay_eps = prioritized_replay_eps
        # Network architecture
        self.num_hidden_layers = num_hidden_layers
        self.num_neurons = num_neurons