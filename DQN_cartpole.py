import gym
import numpy as np
import random
from sklearn import linear_model

def epsilon_greedy(a, eps = 0.1):
    if random.uniform(0,1) < (1-eps):
        return a
    else:
        return random.choice([0,1])


def linear_VFA_MC():
    env = gym.make('CartPole-v0')
    possible_actions = [0,1] #TODO: this shouldnt be hard coded

    num_iters = 10000
    n_features = 4
    alpha = 0.01
    gamma = 0.9

    #initialize weights
    regressor = linear_model.SGDRegressor(verbose = 1)

    for i in range (num_iters):
        state = env.reset()
        episode = []
        t = 0
        done = False
        total_reward =0
        discounted_reward = 0

        X = []
        y = []
        #run episode
        while not done:
            t+=1
            env.render()
            #1 moves to the right, 0 moves to the left
            #action = env.action_space.sample()
            #action = epsilon_greedy(possible_actions[np.argmax([np.dot(state.T,W[action]) for action in possible_actions])])
            if i == 0:
                action = env.action_space.sample()
            else:
                #x.append(action)
                predictions = []
                for action in possible_actions:
                    x = list(state.copy())
                    x.append(action)
                    predictions.append(regressor.predict([x]))
                # print (np.argmax(predictions))
                # print (predictions)
                action = epsilon_greedy(possible_actions[np.argmax(predictions)])

                #action = epsilon_greedy(possible_actions[np.argmax([regressor.predict([x.append(action)]) for action in possible_actions])])
            next_state, reward, done, info = env.step(action)

            Qs = []
            for action_ in possible_actions:
                x = list(next_state.copy())
                x.append(action_)
                if i == 0:
                    Qs.append(0)
                else:
                    Qs.append(regressor.predict([x]))

            y = [reward + gamma*max(Qs)]
            x = list(state.copy())
            x.append(action)

            x = np.array(x).reshape(1, -1)
            y = np.array(y).ravel()

            regressor.partial_fit(x,y)
            #print (regressor.estimator.loss_curve_)


            state = next_state
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break

    env.close()

        #update weights

linear_VFA_MC()
