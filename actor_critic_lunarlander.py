import numpy as np
import gym

env = gum.make("LunarLander-v2")
n_features = 8
n_actions = env.action_space.n


def actor_critic():

    #init everything
    done = False
    s = env.reset()

    while not done:
        env.render()
        a = env.action_space.sample()
        s_prime,r, done, _ = env.step(a)


        s = s_prime
