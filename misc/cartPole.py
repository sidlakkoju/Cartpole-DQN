import gym
env = gym.make('CartPole-v1')

num_steps = 2000

obs = env.reset()

for step in range(num_steps):

    action = env.action_space.sample()

    obs, reward, done, info = env.step(action)

    env.render()

    # time.sleep(0.001)
    
    if done:
        env.reset()
        

env.close()


