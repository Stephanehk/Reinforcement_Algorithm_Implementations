import gym
import numpy as np
import random
import matplotlib.pyplot as plt

def epsilon_greedy(a, eps = 0.1):
    if random.uniform(0,1) < (1-eps):
        return a
    else:
        return random.choice([0,1])


def linear_VFA_MC():
    env = gym.make('CartPole-v0')
    possible_actions = [0,1] #TODO: this shouldnt be hard coded

    num_iters = 1000
    n_features = 4
    alpha = 0.01
    gamma = 0.9

    #initialize weights
    W = [[0 for i in range(n_features)] for j in range (len(possible_actions))]
    reward_arr =  []
    for i in range (num_iters):
        state = env.reset()
        episode = []
        episode_rewards = []
        t = 0
        done = False
        total_reward =0
        # discounted_reward = 0
        #run episode
        while not done:
            t+=1
            env.render()
            #1 moves to the right, 0 moves to the left
            #action = env.action_space.sample()
            action = epsilon_greedy(possible_actions[np.argmax([np.dot(state.T,W[action]) for action in possible_actions])])
            next_state, reward, done, info = env.step(action)
            #----------PROBLEM WITH REWARDS------------------------
            total_reward +=reward
            # discounted_reward += reward*gamma

            #store everything from episode
            episode.append((state,action))
            episode_rewards.append(reward)

            state = next_state
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break

        #plot total reward for episode (just to see if our algorithm is improving)
        reward_arr.append(total_reward)
        plt.plot(reward_arr, color="k")
        plt.pause(0.05)

        #discount reward
        cumulative_reward = 0
        discounted_rewards = [0 for j in range(len(episode_rewards))]
        for j in range(len(episode_rewards)):
            cumulative_reward = cumulative_reward*gamma + episode_rewards[len(episode_rewards)-j-1]
            discounted_rewards[len(episode_rewards)-j-1] = cumulative_reward
        #normalize reward
        discounted_rewards = (discounted_rewards-np.mean(discounted_rewards))/np.std(discounted_rewards)


        #update weights
        prev_reward = 0
        for e in range(len(episode)):
            timestep = episode[e]
            state,action = timestep
            #make prediction
            Q = np.dot(state.T,W[action])
            #get reward
            reward = discounted_rewards[e]
            #calculate the error between reward and prediction
            loss = reward - Q
            #update the weights for each state based on error
            for j in range(len(W[action])):
                #TODO: this is the error function you gotta actually do gradient descent
                #W[action][j] += alpha*np.power(reward - Q,2)*state[j]
                W[action][j] += alpha*loss*state[j]

    plt.show()
    env.close()
    print (W)


        #update weights

linear_VFA_MC()
