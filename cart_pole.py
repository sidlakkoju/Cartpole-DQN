# https://towardsdatascience.com/installing-tensorflow-on-the-m1-mac-410bb36b776

import gym

from collections import deque
import numpy as np
import random
import os

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint


# Definable Parameters
LEARNING_RATE = 0.001       # Learning Rate of the Model
GAMMA = 0.95                # Discount Factor
EPSILON_MAX = 0.30           # Start with only random actions (for maximum exploration) 
EPSILON_MIN = 0.01          # Minimum exploration (Epsilon Greedy)
EPSILON_DECAY = 0.992       # Epsilon decay rate (decays after every taken action)

BATCH_SIZE = 16             # Size of each sample batch used for model training
MEMORY_SIZE = 2000          # Size of memory 

ENVIRONMENT = 'CartPole-v0' # Gym environment

# Checkpoint path and load path kept separate to prevent accidental over-writing of saved model weights
CHECKPOINT_PATH = "cartpole_training_5/cp.ckpt"
LOAD_PATH= "cartpole_training_2/cp.ckpt"            # The saved model was trained for 150 episodes



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
        self.env = gym.make(environment).env
        self.observation_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n

    
    def make_model(self):
        print("Making Model")
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(28, input_shape=(self.observation_space,), activation='tanh'),
            tf.keras.layers.Dense(self.action_space, activation = "linear")
        ])
        self.model.build()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)
        self.model.compile(loss = 'mse', optimizer = self.optimizer)
        self.model.summary()

        # Define Model Checkpoint callback (Save Every Learn Cycle)
        self.checkpoint_path = CHECKPOINT_PATH
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
        self.cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                                save_weights_only=True,
                                                                verbose=0)
        
    
    def action(self, state):
        # Take random action with probability of epsilon
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        
        # Convert List to numpy array and reshape to match proper model input dimensions
        state = np.array(state)                              
        state = state.reshape(1, self.observation_space)

        # Return associated action (0 or 1) with highest q-value
        return np.argmax(self.model.predict(state))
    

    def learn(self):
        # If the memory has less sets than batch_size, do not proceed with learning
        if len(self.memory) < self.batch_size:
            return

        # Batches of 16 random sets from memory used for training
        batch = random.sample(self.memory, self.batch_size)  

        # For each set in batch, add initial observation states and target Q-values to np arrays
        initial_states, target_Qs = np.array([]), np.array([])
        for state, action, reward, next_state, done in batch:
            initial_states = np.append(initial_states, state)
            
            # Reshape state and next_state into a numpy array of proper dimensions        
            state = np.array(state)
            state = state.reshape(1, self.observation_space)         
            next_state = np.array(next_state)
            next_state = next_state.reshape(1, self.observation_space)

            # In terminal cases, target is just reward
            if done:    
                target = reward 

            # Otherwise -> target = reward + GAMMA * maximumQ(Q(next_state))
            else :
                target = reward + (self.discount * np.amax(self.model.predict(next_state)))
            
            # Replace Q-val of the action taken with target, and add to target_Q's
            target_Q = self.model.predict(state)                                         
            target_Q[0][action] = target
            target_Qs = np.append(target_Qs, target_Q)

        # Reshape initial_states and target_Qs to proper dimensions for training
        initial_states = initial_states.reshape((self.batch_size, self.observation_space))
        target_Qs = target_Qs.reshape((self.batch_size, self.action_space))

        # Train Model, the callback saves weights to specified callback path. verbose = 1 training status to console
        self.model.fit(initial_states, target_Qs, callbacks=[self.cp_callback], verbose=1)
            
        # Decay Exploration Rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    # Function to add [self, state, action, reward, next_state, done] to memory dequeue
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    # Function to retrieve saved model weights from given load path in definable paramters.
    def load_model(self):
        self.model.load_weights(LOAD_PATH)
    



    # ---------------- END OF AGENT CLASS ---------------- # 
    
                

# The main function. Trains model.
if __name__ == "__main__":

    # Create a DQN agent. 
    agent = agent()
    agent.load_model()      # Uncomment if you'd like to resume training from a previously saved model. Make sure to update LOAD_PATH in definable parameters.
    episode_counter = 0

    for episode in range(2000):
        state = agent.env.reset()

        for t in range(500):

            # agent.env.render()    # Uncomment if you want to see env as the agent trains

            # Get action for given environment state
            action = agent.action(state)

            # Pass action to environment. Get next_state, associated reward, if done, and additional info
            next_state, reward, done, info = agent.env.step(action)

            # Punish agent if he/she looses
            # reward = reward if not done else -1   

            # Add to memory dequeue
            agent.remember(state, action, reward, next_state, done)

            # Train Model
            agent.learn()

            # Make next_state the current state
            state = next_state

            # TERMINAL CONDITION, YOUR DONE BOI, reset env and TRY AGAIN
            if done:
                episode_counter += 1
                print("Episode Number: " + str(episode_counter) + ". Episode finished after {} timesteps".format(t+1))
                break

    agent.env.close()
