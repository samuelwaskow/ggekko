import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.backend_config import epsilon

from agent import DuelingDDQNAgent
from env_daytrade import DaytradeEnvironment
from utils import plot_daytrade

tf.keras.backend.set_floatx('float64')

if __name__ == '__main__':

    symbol = sys.argv[1]
    save_model = 10
    update_target = 10
    window = 2
    checkpoint = f'../checkpoint/{symbol}'
    
    print(f'System parameters - symbol [{symbol}] update_target [{update_target}] save_model [{save_model}] checkpoint [${checkpoint}]')
    env = DaytradeEnvironment(f'../data/{symbol}.csv', 
                            window_size=window,
                            features=['PRICE', 'MACD_MAIN','MACD_SIGNAL','AWESOME','STO_MAIN','STO_SIGNAL'])
    agent = DuelingDDQNAgent(
                    input_dims=env.get_observation_space(), 
                    n_actions=env.get_action_space(),
                    fname=checkpoint)
    
    agent.load_model()
    scores = []
    for i in range(env.end_step):
        scores.append([])
    start = 0

    for j in range(40000001):
        start = np.random.randint(0, env.end_step - 1)
        done = False
        observation = env.reset(start)
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done = env.step(action)
            agent.store_transition(observation, action, reward, observation_, done)
            observation = observation_
            agent.learn()
            
        agent.update_epsilon()       
        scores[start].append(env.balance)

        if j > 1 and j % update_target == 0:
            agent.align_networks()
        if j > 1 and j % save_model == 0:
            agent.save_model()
            filename = f'daytrader_{symbol}_{start}.png'
            plot_daytrade(scores[start], filename)
    
        print(
            'eps', j,
            'reward %.6f' % env.balance,
            'steps', env.steps,  
            'start', start,
            'epsilon %.2f' % agent.epsilon)    
        
        
        
    
    