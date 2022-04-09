import gym
from cart_pole import agent
env = gym.make('CartPole-v0')


agent = agent()
agent.load_model()

for i_episode in range(300):
    observation = agent.env.reset()

    while True:
        agent.env.render()
        #action = env.action_space.sample()
        action = agent.action(observation)                    # get action
        observation, reward, done, info = agent.env.step(action)  # take action
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
  
agent.env.close()