import numpy as np

class DuelingDDQNReplayBuffer(object):

    #
    # Constructor
    #
    def __init__(self, max_size, input_shape):
        self.max_size = max_size
        self.input_shape = input_shape
        self.reset()
    
    #
    # Stores the tuple S,A,R,S'
    #
    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        self.mem_cntr += 1


    #
    # Sample the buffer
    #
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        dones = self.terminal_memory[batch]
        return states, actions, rewards, states_, dones

    #
    # Refresh
    #
    def reset(self):
        self.mem_size = self.max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *self.input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *self.input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.int32)
        
        
