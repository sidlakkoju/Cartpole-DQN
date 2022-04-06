import gym
from dqn import dqn
env = gym.make('CartPole-v0')


dqn = dqn(env.observation_space.shape[0], env.action_space.n)
dqn.load_model()

for i_episode in range(300):
    observation = env.reset()

    for t in range(200):
        env.render()
        #action = env.action_space.sample()
        action = dqn.action(observation)                    # get action
        observation, reward, done, info = env.step(action)  # take action
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
  
env.close()