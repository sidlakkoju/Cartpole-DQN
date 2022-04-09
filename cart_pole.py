import gym

from collections import deque
import numpy as np
import random
import os

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint


#Definable Parameters
LEARNING_RATE = 0.001       # Learning Rate of the Model
GAMMA = 0.95                # Discount Factor
EPSILON_MAX = 1.0           # Start with only random actions (for maximum exploration) 
EPSILON_MIN = 0.01          # Minimum exploration (Epsilon Greedy)
EPSILON_DECAY = 0.995       # Epsilon decay rate (decays after every taken action)

BATCH_SIZE = 16             # Size of each sample batch used for model training
MEMORY_SIZE = 2000          # Size of memory 

ENVIRONMENT = 'CartPole-v0'
CHECKPOINT_PATH = "cartpole_training1/cp.ckpt"



class agent:

    def __init__(self):
        # Initialize Parameters
        self.learning_rate = LEARNING_RATE
        self.discount = GAMMA
        self.epsilon = EPSILON_MAX
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.batch_size = BATCH_SIZE
        self.memory = deque(maxlen = MEMORY_SIZE)

        # Make Environment and Model
        self.make_env(ENVIRONMENT)
        self.make_model()

    
    def make_env(self, environment):
        print("Making Environment")
        self.env = gym.make(environment)
        self.observation_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n

    
    def make_model(self):
        print("Making Model")
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(24, input_shape=(self.observation_space,), activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(self.action_space, activation = "linear")
        ])
        self.model.build()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)
        self.model.compile(loss = 'mse', optimizer = self.optimizer)
        self.model.summary()

        # Define Model Checkpoint callback (Save Every 5 learn() cycles)
        self.checkpoint_path = CHECKPOINT_PATH
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
        self.cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                                save_weights_only=True,
                                                                save_freq = self.batch_size*5,
                                                                verbose=0)
        
    
    def action(self, state):
        if np.random.rand() <= self.epsilon:
            random.randrange(self.action_space)
        
        # Convert List to numpy array and reshape to match proper model input dimensions
        state = np.array(state)                              
        state = state.reshape(1, self.observation_space)

        return np.argmax(self.model.predict(state))
    

    def learn(self):
        # If the memory has less sets than batch_size, do not proceed with learning
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)  

        

        # Get random training batch from memory
        for state, action, reward, next_state, done in batch:

            # Reshape state and next_state into a numpy array of proper dimensions
            state = np.array(state)  
            state = state.reshape(1, self.observation_space) 
            next_state = np.array(next_state)
            next_state = next_state.reshape(1, self.observation_space)

            # In terminal cases, target is just reward
            target = reward 
                                                    

            # Otherwise -> target = reward + GAMMA * maximumQ(Q(next_state))
            if not done:
                target = reward + (self.discount * np.amax(self.model.predict(next_state)))
            
            target_Q = self.model.predict(state)

            # Replace Q-val of the action taken with target, then fit                                               
            target_Q[0][action] = target

            self.model.fit(state, target_Q, callbacks=[self.cp_callback], verbose=0)
            
            # Decay Exploration Rate
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
    

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    

    def load_model(self):
        self.model.load_weights(self.checkpoint_path)
    
    # ---------------- END OF AGENT CLASS ---------------- # 



def runModel():
    agent = agent()
    agent.load_model()
    for episode in range(300):
        observation = agent.env.reset()
        t = 0
        while True:
            t+=1
            agent.env.render()
            #action = env.action_space.sample()
            action = agent.action(observation)                    # get action
            observation, reward, done, info = agent.env.step(action)  # take action
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
            

if __name__ == "__main__":

    agent = agent()
    agent.load_model()
    episode_counter = 0

    for episode in range(2000):
        state = agent.env.reset()

        for t in range(500):
            agent.env.render() # Uncomment if you want to see env as the agent trains
            action = agent.action(state)
            next_state, reward, done, info = agent.env.step(action)

            # Punish agent if he/she looses
            reward = reward if not done else -10   

            agent.remember(state, action, reward, next_state, done)
            agent.learn()
            state = next_state

            if done:
                episode_counter += 1
                print("Episode Number: " + str(episode_counter) + ". Episode finished after {} timesteps".format(t+1))
                break

    agent.env.close()
