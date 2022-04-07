import gym
from dqn import dqn
env = gym.make('CartPole-v0')


dqn = dqn(env.observation_space.shape[0], env.action_space.n)
dqn.load_model()

episodeCounter = 0

for i_episode in range(2000):
    observation = env.reset()

    for t in range(500):
        old_obs = observation                               # store observation
        action = dqn.action(observation)                    # get action
        observation, reward, done, info = env.step(action)  # take action
        dqn.remember(old_obs, action, reward, observation, done)
        dqn.learn()
        if done:
            episodeCounter+=1

            print("Episode Number: " + str(episodeCounter) + ". Episode finished after {} timesteps".format(t+1))
            break
  
env.close()