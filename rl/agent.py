import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from replay_buffer import DuelingDDQNReplayBuffer
from network import DuelingDeepQNetwork

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class DuelingDDQNAgent(object):

    #
    # Constructor
    #
    def __init__(self, 
                n_actions, 
                input_dims, 
                fname,
                alpha=0.0005,
                gamma=0.99, 
                batch_size=32, 
                epsilon=1, 
                epsilon_dec=0.99, 
                epsilon_end=0.1, 
                mem_size=1000000, 
                fc1_dims=256,
                fc2_dims=256):    
        self.action_space = [i for i in range(n_actions)]
        print(f'Agent - state [{input_dims}] actions [{self.action_space}] file [{fname}]')
        self.gamma = gamma
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname
        self.learn_step_counter = 0
        self.memory = DuelingDDQNReplayBuffer(mem_size, input_dims)

        self.q_network = DuelingDeepQNetwork(n_actions, fc1_dims, fc2_dims)
        self.target_network = DuelingDeepQNetwork(n_actions, fc1_dims, fc2_dims)

        self.q_network.compile(optimizer=Adam(learning_rate=alpha), loss='mse')
        self.target_network.compile(optimizer=Adam(learning_rate=alpha), loss='mse')


    #
    # Store S,A,R,S' in the memory (replay buffer)
    #
    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    #
    # Choose an action based on the state
    #
    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            action = self.advantage(observation)
        return action

    #
    # Chooses an action based on its most value
    # Needs to be created since the method is not loaded by the model
    # Forward Propagation ;)
    #
    def advantage(self, observation):
        state = np.array([observation])
        x = self.q_network.dense1(state)
        x = self.q_network.dense2(x)
        actions = self.q_network.A(x)
        return tf.math.argmax(actions, axis=1).numpy()[0]

    #
    # Learn itself
    #
    def learn(self):
        if self.memory.mem_cntr < self.batch_size: # Do not learn util the memory has at least the batch size
            return

        states, actions, rewards, states_, dones = self.memory.sample_buffer(self.batch_size)
        
        q_pred = self.q_network(states)
        q_next = self.target_network(states_)
        q_target = q_pred.numpy()
        
        max_actions = tf.math.argmax(self.q_network(states_), axis=1)

        for i, terminal in enumerate(dones):
            max_next = q_next[i, max_actions[i]]
            done = (1 - int(dones[i]))
            q_target[i, actions[i]] = rewards[i] + self.gamma * max_next * done

        self.q_network.train_on_batch(states, q_target)
        self.learn_step_counter += 1


    #
    # Align the target network with the Q network
    #
    def align_networks(self):
        self.target_network.set_weights(self.q_network.get_weights())

    #
    # Decay the epsilon value
    #
    def update_epsilon(self):
        self.epsilon = self.epsilon * self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min

    #
    # Store the model
    #
    def save_model(self):
        self.q_network.save(self.model_file)

    #
    # Load the stored model
    #
    def load_model(self):
        #if os.path.isfile(self.model_file):
            self.q_network = tf.keras.models.load_model(self.model_file)
            self.target_network = tf.keras.models.load_model(self.model_file)
        #else:
        #    print(f'Model doesn\'t exist: {self.model_file}')
