import numpy as np
import gym
import matplotlib.pyplot as plt
import random
import warnings

env = gym.make('CartPole-v0')
n_actions = env.action_space.n
n_features = 4
#warnings.filterwarnings('error')

def gradient_checkinb(s, theta, e = 0.001):
    theta = theta.unravel()
    grad_approx = [None for i in range (len(theta))]
    for i in range (len(theta)):
        theta_plus = theta
        theta_plus[i] += e
        theta_minus = theta
        theta_minus[i] -= e
        grad_approx[i] = (follow_policy(theta_plus, s, False) - follow_policy(theta_minus, s, False))/(2*e)

def phi (s,a):
    phi = np.zeros([n_actions,n_features])
    phi[a] = s
    #phi = phi.flatten()
    return phi

def follow_policy(thetas, s, return_action):

    #possible_actions = [np.dot(s.T,thetas[a]) for a in range(n_actions)]
    # print (thetas)
    # print (thetas[:1])
    possible_actions = [np.dot(s.T,thetas.T[a]) for a in range(n_actions)]
    softmax = np.exp(possible_actions)/np.exp(sum(possible_actions))

    #TODO: Problem could be here - should chose action based on probability instead of always chosing the action witht the highest probability
    if return_action:
        return np.argmax(softmax)
    else:
        return softmax

def MC_policy_gradient ():
    num_iters = 1000
    gamma = 0.99
    alpha = 0.01
    #setup weights
    #thetas = [[random.uniform(0, 1) for i in range(n_features)] for j in range (n_actions)]
    thetas = np.array([[random.uniform(0, 1) for i in range(n_actions)] for j in range (n_features)])

    total_rewards_arr = []
    avg_reward_arr = [0]

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
        while not done and t < 300:
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
                moving_avg = (total_reward - avg_reward_arr[-1]) * (2/(len(avg_reward_arr) +1)) + avg_reward_arr[-1]
                avg_reward_arr.append(moving_avg)

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
            #get softmax prob
            policy_probs = follow_policy(thetas, s_t, False)

            #format phi
            phi_ = phi (s_t,a_t)
            softmax_update = phi_ - sum([phi(s_t,a)*policy_probs[a] for a in range(n_actions)])
            #calculate baseline (average reward from s onwards)
            b = np.mean(discounted_rewards[list(discounted_rewards).index(r):])
            thetas+= alpha*(r-b)*softmax_update.T

        plt.plot(avg_reward_arr, color = "g")
        plt.pause(0.05)
    plt.show()

MC_policy_gradient ()
