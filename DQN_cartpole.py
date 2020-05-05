import gym
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import matplotlib.pyplot as plt

possible_actions = [0,1]
gamma = 0.9
eps = 0.1

def reservoir_sample(arr, k):
    reservoir = []
    for i in range (len(arr)):
        if len(reservoir) < k:
            reservoir.append(arr[i])
        else:
            j = int(random.uniform(0,i))
            if j < k:
                #print (j)
                reservoir[j] = arr[i]
    return reservoir

def get_best_action (model, s):
    #return possible_actions[np.argmax(np.dot(s.T,Q[a]) for a in possible_actions)]
    next_action_values = []
    for a in possible_actions:
        x = list(s.copy())
        x.append(a)
        #x = np.reshape(x, (5,1))
        x = np.array([x])
        #print (x.shape)
        #x = np.array([[x_] for x_ in x])
        pred = model.predict(x)
        next_action_values.append(pred[0][0])

    return possible_actions[np.argmax(next_action_values)]

def epsilon_greedy(a):
    if random.uniform(0,1) < (1-eps):
        return a
    else:
        return random.choice(possible_actions)

def Q_learn ():
    env = gym.make("CartPole-v0")

    #memory replay array
    D = []

    num_episodes = 10000
    c = 0

    #setup neural network
    opt = SGD(lr=0.01)

    model = Sequential()
    model.add(Dense(units=12, activation='relu', input_dim=5))
    model.add(Dense(units=8, activation='relu'))
    model.add(Dense(units=1, activation='softmax'))
    model.compile(loss='mean_squared_error',optimizer=opt, metrics = ["mse"])

    #setup target neural network
    target_model = Sequential()
    target_model.add(Dense(units=12, activation='relu', input_dim=5))
    target_model.add(Dense(units=8, activation='relu'))
    target_model.add(Dense(units=1, activation='softmax'))
    target_model.compile(loss='mean_squared_error',optimizer=opt, metrics = ["mse"])

    reward_arr = []
    mse_arr = []
    for i in range (num_episodes):
        s = env.reset()
        done = False
        total_reward = 0
        t = 0

        while not done:
            env.render()
            t+=1

            #sample 64 (s,a,r,s_prime) pairs to train NN on
            if len (D)> 64:
                a = epsilon_greedy(get_best_action (model, s))
                #run single instance of episode
                s_prime, reward, done, info = env.step(a)
                total_reward += reward
                D.append((s,a,total_reward,s_prime))


                sample =reservoir_sample(D,64)
                #build X and y
                X = []
                y = []
                for episode in sample:
                    s,a,reward,s_prime = episode

                    x = list(s.copy())
                    x.append(a)
                    X.append(x)

                    #build y using best predicted s,a pair
                    possible_preds = []
                    for a in possible_actions:
                        p_x = list(s.copy())
                        p_x.append(a)
                        p_x = np.array([p_x])
                        #TODO: WE ALL KNOW THERE IS GONNA BE ANOTHER REWARD ERROR!!
                        pred = reward + gamma*target_model.predict(p_x)
                        possible_preds.append(pred[0][0])
                    #print ("possible preds shape: " + str(np.array(possible_preds).shape))
                    y.append([max(possible_preds)])
                    #print (max(possible_preds))

                #train neural network
                X = np.array(X)
                y = np.array(y)
                #history = model.train_on_batch(X,y)
                history = model.fit(X,y,verbose=0)

                # error = model.evaluate(X, y,verbose=0)
                # #print ("model error: " + str(error))

                # plt.figure(0)
                mse_arr.extend(history.history["mean_squared_error"])
                # plt.plot(mse_arr, color="r")
                # plt.pause(0.05)
            else:
                a = epsilon_greedy(random.choice(possible_actions))
                #run single instance of episode
                s_prime, reward, done, info = env.step(a)
                total_reward += reward
                D.append((s,a,total_reward,s_prime))


            s = s_prime
            #transfer weights to target Q
            c+=1
            #TODO: idk after how many steps should the weights be transfered
            if c == 128:
                #print ("tranfered weights...")
                target_model.set_weights(model.get_weights())
                c = 0
        reward_arr.append(total_reward)

        plt.figure(1)
        plt.plot(reward_arr, color="k")
        plt.pause(0.05)

        plt.figure(0)
        plt.plot(mse_arr, color="r")
        plt.pause(0.05)

        if done:
            print("Episode finished after {} timesteps".format(t+1))


    plt.show()
    env.close()
Q_learn()
