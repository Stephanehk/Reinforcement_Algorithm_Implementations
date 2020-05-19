import numpy as np
import gym
import matplotlib.pyplot as plt
import random
import warnings
from scipy.special import softmax


env = gym.make('CartPole-v0')
n_actions = env.action_space.n
n_features = 8
warnings.filterwarnings('error')

def follow_policy(thetas, s, return_action):

    #possible_actions = [np.dot(s.T,thetas[a]) for a in range(n_actions)]
#    possible_actions = [np.dot(thetas[a],s.T) for a in range(n_actions)]
    possible_actions = [None for a in range (n_actions)]
    for a in range(n_actions):
        s_copy = list(s.copy())
        s_copy.append(a)
        s_copy = np.array(s_copy)
        possible_actions[a] = np.dot(thetas,s_copy.T)

    softmax = np.exp(possible_actions)/np.exp(sum(possible_actions))
    #softmax_vals= softmax(possible_actions)
    # print (possible_actions)
    # print (softmax)
    # print ("\n")

    #TODO: Problem could be here - should chose action based on probability instead of always chosing the action witht the highest probability
    if return_action:
        return np.argmax(softmax)
    else:
        return softmax



def MC_policy_gradient ():
    num_iters = 1000
    gamma = 0.9
    alpha = 0.001
    #setup weights
    thetas = [random.uniform(0, 1) for i in range(n_features + 1)]
    total_rewards_arr = []

    for i in range(num_iters):

        #init everything
        s = env.reset()
        done = False
        t = 0
        discounted_reward = 0
        episode = []
        episode_rewards = []
        total_reward = 0

        #run an episode
        while not done:
            env.render()
            t+=1
            a = follow_policy(thetas, s, True)

            s_prime, reward, done, info = env.step(a)

            #handle rewards
            total_reward += reward
            #discounted_reward += np.power(gamma,t)*reward
            episode_rewards.append(reward)
            episode.append((s,a,s_prime))

            if done:
                print("Episode finished after {} timesteps".format(t+1))
                total_rewards_arr.append(total_reward)
                env.close()
                break
            else:
                s = s_prime

        #generate cumulative discounted reward
        discounted_rewards = [None for i in  range (len(episode_rewards))]
        r = 0
        for j in range (len(episode_rewards)):
            r = (gamma * r) + episode_rewards[len(episode_rewards)-j-1]
            discounted_rewards[len(episode_rewards)-j-1] = r

        #normalize discounted rewards
        discounted_rewards = (discounted_rewards - np.mean(discounted_rewards))/np.std(discounted_rewards)

        #iterate through episode
        for step, r in zip(episode,discounted_rewards):
            s_t,a_t,s_prime_t = step


            #PROBLEM: this is not the correct update rule?
            #log softmax update
            #print ([np.dot(s_t.T,thetas[a]) for a in range(n_actions)])

            #softmax_update = np.dot(s_t.T,thetas[a_t]) - sum([np.dot(s_t.T,thetas[a]) for a in range(n_actions)])
            policy_probs = follow_policy(thetas, s, False)
            #softmax_update = np.dot(s_t.T,thetas[a_t]) - sum([np.dot(s_t.T,thetas[a]) * policy_probs[a] for a in range(n_actions)])

            #calculate eligibility vector
            sum = 0
            for a in range(n_actions):
                s_t_copy = list(s_t.copy())
                s_t_copy.append(a)
                sum += np.dot(policy_probs[a],np.array(s_t_copy).T)
            s_t_copy = list(s_t.copy())
            s_t_copy.append(a_t)
            s_t_copy = np.array(s_t_copy).T
            softmax_update = s_t_copy - sum

            #print (np.dot(s_t.T,thetas[0]))
            thetas+= alpha*r*softmax_update
            # print ("thetas: ")
            # print (thetas)
            # print ("------------------------------------------------")
    #    print (thetas)
            #print (thetas)
        plt.plot(total_rewards_arr, color = "k")
        plt.pause(0.05)
    plt.show()

MC_policy_gradient ()
