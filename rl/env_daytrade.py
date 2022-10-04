
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow._api.v2 import data

from tensorflow.python.ops.gen_dataset_ops import window_dataset


class DaytradeEnvironment(object):

    POSITION_INDEX = 0
    CLOSE_INDEX = 1

    #
    # Constructor
    #
    # 0 = Position
    # 1 = End State
    # 2 = Close Price
    #

    def __init__(self,
                 filename,
                 window_size,
                 features,
                 ):
        print(
            f'DaytradeEnvironment - filename [{filename}] window_size [{window_size}]')
        self.window_size = window_size
        self.features = features
        self.position = 0
        self.current_step = 0
        self.end_step = 0
        self.states = []
        self.filename = filename

        if filename != None:
            df = pd.read_csv(filename, parse_dates=['DATETIME'], low_memory=False)
            features = self.get_dataset(df)

            dataset = self.get_window_dataset(features.to_numpy(dtype=float))
            for w in dataset:
                self.states.append(self.normalize_window(w))

            self.end_step = len(self.states)
        else:
            w = np.zeros((self.window_size, len(self.features)), dtype=np.float32)
            self.states.append(w.reshape(self.window_size * len(self.features)))

    #
    # Get the day dataset
    #
    def get_dataset(self, day):
        features = day[self.features]
        features.index = day['DATETIME']
        return features

    #
    # Generate time windows sequence-to-sequence
    # Normalize having the first element from the windows as basis
    #
    def normalize_window(self, window):

        #print(window)
        out = window.reshape(self.window_size * len(self.features))
        #out[0] = out[8]
        #out = np.delete(out, [8])
        out = np.insert(out, self.POSITION_INDEX, 0.0) 
        print(out)
        print('#########')
        return out

    #
    # Generate time windows sequence-to-sequence
    # Transform the dataset into a dataset containing windows
    # Transform each window into a tensor avoiding a second loop to iterate over the elements
    #
    def get_window_dataset(self, series):
        ds = tf.data.Dataset.from_tensor_slices(series)
        ds = ds.window(self.window_size, shift=1, drop_remainder=True)
        ds = ds.flat_map(lambda w: w.batch(self.window_size))
        return ds.as_numpy_iterator()

    #
    # Populate the state values with data
    #
    def get_obs(self):
        obs = np.copy(self.states[self.current_step])
        obs[self.POSITION_INDEX] = self.position
        #print(obs)
        return obs

    #
    # Reset the environment
    #
    def reset(self, step):
        self.position = 0
        self.balance = 0
        self.current_step = step
        self.steps = 1
        self.done = False
        obs = self.get_obs()
        if len(obs) > self.CLOSE_INDEX:
            self.close = obs[self.CLOSE_INDEX]
        obs = self.clear_state(obs)
        return obs

    #
    # Get the data related to the step
    #
    def step(self, action):

        obs = self.get_obs()

        self.close = obs[self.CLOSE_INDEX]

        r = self.get_reward(action)

        obs = self.clear_state(obs)

        if self.end():
            self.done = True
            r = self.balance
        else:  
            self.current_step += 1
            self.steps += 1
        
        return obs, r, self.done

    #
    # Return the agent reward
    #
    def get_reward(self, action):

        r = 0
        if self.position != 0:
            if self.position == -0.1:
                r = 1 - (self.close / self.start_close)
                if action == 2:
                    self.position = 0
            else:
                r = (self.close / self.start_close) - 1
                if action == 0:
                    self.position = 0
        else:     
            if action == 0:
                self.position = -0.1
                self.start_close = self.close
            elif action == 2:
                self.position = 0.1
                self.start_close = self.close
        
        self.balance += r
        return r

    #
    # Space
    #

    def get_observation_space(self):
        return self.reset(0).shape

    #
    # Removes extra zeros
    #
    def clear_state(self, obs):
        return np.delete(obs, [self.CLOSE_INDEX])

    #
    # Number of actions
    #
    def get_action_space(self):
        return 3

    #
    # Check if it is the end of an episode
    #
    def end(self):
        return self.current_step >= self.end_step - 1