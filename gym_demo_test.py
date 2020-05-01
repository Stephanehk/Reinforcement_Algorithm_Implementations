import gym
import numpy as np


env = gym.make('CartPole-v0')
observation = env.reset()
done = False

while not done:
    env.render()
    #print(observation)
    #1 moves to the right, 0 moves to the left
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    print (reward)
    if done:
        print("Episode finished after {} timesteps".format(t+1))
        break
env.close()
