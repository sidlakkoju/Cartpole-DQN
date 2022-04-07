import gym

import numpy as np
import random
import os

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam # - Works
from tensorflow.keras.callbacks import ModelCheckpoint

# Refer to link bellow to make video and save weights 
# https://towardsdatascience.com/deep-reinforcement-learning-build-a-deep-q-network-dqn-to-play-cartpole-with-tensorflow-2-and-gym-8e105744b998


# Definable Parameters
GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 16

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.10
EXPLORATION_DECAY = 0.99

class dqn:

    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        self.exploration_rate = EXPLORATION_MAX
        self.memory = []
        self.make_model()
    

    def remember(self, state, action, reward, next_state, done):
        if len(self.memory) >= MEMORY_SIZE:
            self.memory = self.memory[-1:]
        self.memory.append([state, action, reward, next_state, done])

    def action(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        
        state = np.array([state])
        state = state.reshape((1, self.observation_space))
        q_values = self.model.predict(state)
        return np.argmax(q_values)
    
    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        batch = random.sample(self.memory, BATCH_SIZE)
        
        for state, action, reward, next_state, done in batch:
        
            if not done:
                next_state = np.array([next_state])
                next_state.reshape((1, self.observation_space))

                expectedQ = reward + GAMMA*self.model.predict(next_state)
            else :
                expectedQ = reward
                expectedQ = np.array([expectedQ]*2)
                expectedQ = expectedQ.reshape((1, self.action_space))
                
            state = np.array([state])
            state = state.reshape((1, self.observation_space))

            self.model.fit(state, expectedQ,callbacks=[self.cp_callback], verbose=0)
            #self.model.fit(state, expectedQ, verbose=0)


        if self.exploration_rate > EXPLORATION_MIN:
            self.exploration_rate = self.exploration_rate*EXPLORATION_DECAY
       
        
    
    def make_model(self):
        self.model = keras.Sequential()
        self.model.add(layers.Dense(24, input_shape=(self.observation_space,), activation="relu"))
        self.model.add(layers.Dense(48, activation="relu"))
        self.model.add(layers.Dense(24, activation="relu"))
        self.model.add(layers.Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(learning_rate=LEARNING_RATE))

        
        self.checkpoint_path = "training_2/cp.ckpt"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
        self.cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                 save_weights_only=True,
                                                 save_freq=5*BATCH_SIZE,
                                                 verbose=0)
        
    
    def load_model(self):
        self.model.load_weights(self.checkpoint_path)


  



    

