import numpy as np
import gym
import random
import matplotlib.pyplot as plt
import warnings
#warnings.filterwarnings('error')

#env = gym.make("LunarLander-v2")
env = gym.make("CartPole-v0")
n_features = 4
n_actions = env.action_space.n

def follow_policy(thetas, s, return_action):
    possible_actions = [np.dot(s.T,thetas.T[a]) for a in range(n_actions)]
    softmax = np.exp(possible_actions)/sum(np.exp(possible_actions))

    #TODO: Problem could be here - should chose action based on probability instead of always chosing the action witht the highest probability
    if return_action:
        return np.random.choice(n_actions,p=softmax)
    else:
        return softmax

def phi (s,a):
    phi = np.zeros([n_actions,n_features])
    phi[a] = s
    #phi = phi.flatten()
    return phi

def actor_critic():
    #init everything
    gamma = 0.99
    alpha_theta = 0.01
    alpha_w = 0.01
    num_iters = 10000

    #init weights
    thetas = np.array([[random.uniform(0, 1) for i in range(n_actions)] for j in range (n_features)])
    W = np.array([[random.uniform(0, 1) for i in range(n_features)] for j in range (n_actions)])
    avg_reward_arr = [0]

    #debugging stuff
    first_episode_reward = None
    for i in range(num_iters):
        done = False
        s = env.reset()
        a = follow_policy(thetas, s, True)
        t = 0
        total_reward = 0
        while not done:
            #env.render()
            t+=1
            s_prime,r, done, _ = env.step(a)
            total_reward+=r
            #----------------CRITIC-----------------------------------

            if not done:
                a_prime = follow_policy(thetas, s_prime, True)
                delta = r + gamma*np.dot(s_prime.T,W[a_prime]) - np.dot(s.T,W[a])
            else:
                delta = r - np.dot(s.T,W[a])
            #update weight W
            W+=alpha_w*delta*phi(s,a)

            #----------------ACTOR-----------------------------------
            #get policy probabilities
            policy_probs = follow_policy(thetas, s, False)
            #log of policy gradient
            softmax_update = phi (s,a) - sum([phi(s,a_)*policy_probs[a_] for a_ in range(n_actions)])
            #update weight thetas
            thetas+= alpha_theta*np.dot(s.T,W[a])*softmax_update.T

            s = s_prime
            a = a_prime

            if done:
                if i == 0:
                    first_episode_reward = total_reward
                print("Episode finished after {} timesteps".format(t+1))
                moving_avg = (total_reward - avg_reward_arr[-1]) * (2/(len(avg_reward_arr) +1)) + avg_reward_arr[-1]
                avg_reward_arr.append(moving_avg)
                env.close()
                break

        plt.axhline(y=first_episode_reward, color='g', linestyle='-')
        plt.plot(avg_reward_arr, color = "k")
        plt.pause(0.05)
    plt.show()

actor_critic()
