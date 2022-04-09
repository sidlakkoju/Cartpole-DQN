import gym
from cart_pole import agent


agent = agent()
agent.load_model()
agent.epsilon = 0.50

for i_episode in range(300):
    observation = agent.env.reset()
    t=0
    while True:
        t+=1
        agent.env.render()
        action = agent.action(observation)                          # get action
        observation, reward, done, info = agent.env.step(action)    # take action
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
  
agent.env.close()
