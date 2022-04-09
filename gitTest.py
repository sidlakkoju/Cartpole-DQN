from collections import deque
import tensorflow as tf
import pandas as pd
import numpy as np
import random
import time
import gym
import os


class agent:
    def __init__(self):
        #Formula
            #NAIVE Q LEARNING
                #Q(s,a) = (1 - Gamma) * Q(s,a) + Gamma*(reward + Alpha * argmax (Q(s', a')))
                #Alpha = Learning Rate
                #Gamma = Discount Factor
                #argmax = Deep Learning prediction
        
        self.discount = 0.95
        self.epsilon = 1.0 
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        self.batchSize = 16
        self.done = False      
        
        self.memory = deque(maxlen=2000)
        
        self.actions = ['left', 'right']
        self.qTable = pd.DataFrame(columns=self.actions, dtype = np.float64)
        
        self.checkPointPath = "DQNcartpole2/cp.ckpt"
        self.checkpoint_dir = os.path.dirname(self.checkPointPath)
        self.model = self.defineModel()
        try:
            self.model = self.model.load_weights(self.checkPointPath)
            print("MODEL LOADED")
        except:
            pass

    def defineModel(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(24, input_dim = 4, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(2, activation = "linear")
        ])
        self.model.build()
        self.optimizer = tf.keras.optimizers.Adam(lr = self.learning_rate)
        self.model.compile(loss = 'mse', optimizer = self.optimizer)
        self.model.summary()
        
        return self.model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def chooseAction(self, state):
        if np.random.rand() <= self.epsilon:
            print("RANDOM ACTION")
            return np.random.choice([0,1])
        #print("CHOSEN ACTION")
        return np.argmax(self.model.predict(state))
        
    def learn(self):
        batch = random.sample(self.memory, self.batchSize)
        model = self.model
        
        for state, action, reward, nextState, done in batch:
            target = reward
            
            if not done:
                #print(nextState)
                target = ( reward + self.discount * np.amax( model.predict(nextState) ))
                
            target_f = model.predict(state)
            
            target_f[0][action] = target
            
            self.model.fit(state, target_f, epochs=1, verbose=0)
            
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
    
    def saveWeights(self):
        cp_callback = tf.keras.callbacks.ModelCheckpoint(self.checkPointPath,
                                                                save_weights_only=True,
                                                                verbose=1) 
        return cp_callback 
                
    def runENV(self):
        env = gym.make('CartPole-v1')
        reward = 0
        
        for episode in range(1000):
            state = env.reset()
            state = np.reshape(state, [1, 4])
            
            for steps in range(500):
                env.render()
                print('Episode %s: total_steps = %s Reward = %s' % (episode+1, steps, reward))
                action = self.chooseAction(state)
                
                nextState, reward, done, _ = env.step(int(action))
                
                reward = reward if not done else -10
                state = np.reshape(state, [1, 4])
                nextState = np.reshape(nextState, [1, 4])
                
                self.remember(state, int(action), reward, nextState, done)
                state = nextState
                
                if done:
                    break
                
                if len(self.memory) >= self.batchSize:
                    self.learn()
                              
        
a = agent()

a.runENV()        
           