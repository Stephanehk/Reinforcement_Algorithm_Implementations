import gym
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense,Activation,BatchNormalization
from keras.optimizers import SGD,Adam
from keras.models import model_from_json
import matplotlib.pyplot as plt

possible_actions = [0,1]
gamma = 0.9

# eps = 0.1
# Exploration parameters
explore_start = 1.0            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability
decay_rate = 0.0001            # exponential decay rate for exploration prob


def reservoir_sample(arr, k):
    reservoir = []
    for i in range (len(arr)):
        if len(reservoir) < k:
            reservoir.append(arr[i])
        else:
            j = int(random.uniform(0,i))
            if j < k:
                reservoir[j] = arr[i]
    return reservoir

def get_best_action (model, s):
    s = np.array(s)
    s = s.reshape(1,4)
    action_values = model.predict(s)
    return np.argmax(action_values[0])


def epsilon_greedy(a,explore_p):
    if explore_p > random.uniform(0,1):
        return a
    else:
        return random.choice(possible_actions)

def Q_learn ():

    def creat_model():
        #opt = SGD(lr=0.001)
        opt = Adam(lr=0.001)
        #try out different configs of batch norm
        model = Sequential()
        model.add(Dense(units=12, input_dim=4, activation="relu"))
        model.add(Dense(units=8, activation="relu"))
        model.add(Dense(units=2, activation='linear'))
        model.compile(loss='mean_squared_error',optimizer=opt, metrics = ["mse"])
        return model

    env = gym.make("CartPole-v0")

    #memory replay array
    D = []

    num_episodes = 200
    pre_train_num_episodes = 32
    max_steps = 200
    c = 0

    #setup neural network
    model = creat_model()
    #setup target neural network
    target_model = creat_model()


    reward_arr = []
    mse_arr = []
    avg_reward_arr = [0]


    #run epsiodes randomly to generate some training data
    for iter in range (pre_train_num_episodes):
        s = env.reset()
        done = False
        while not done:
            env.render()
            a = epsilon_greedy(random.choice(possible_actions),0.1)
            #run single instance of episode
            s_prime, reward, done, info = env.step(a)
            D.append((s,a,reward,s_prime,done))
            s = s_prime

    #start running episodes and actually improving
    for iter in range (num_episodes):
        s = env.reset()
        done = False
        total_reward = 0
        t = 0

        while t < max_steps:
            env.render()
            t+=1
            explore_p = explore_stop + (explore_start - explore_stop)*np.exp(-decay_rate*t)
            #sample 64 (s,a,r,s_prime) pairs to train NN on

            a = epsilon_greedy(get_best_action (model, s),explore_p)
            #run single instance of episode
            s_prime, reward, done, info = env.step(a)
            total_reward += reward

            if done:
                print("Episode " + str(iter) + " finished after {} timesteps".format(t+1))
                D.append((s,a,reward,s_prime,done))
                t = max_steps
            else:
                D.append((s,a,reward,s_prime,done))
                s = s_prime


            #--------memory replay--------------------------------------------------------------------------------------------
            sample =reservoir_sample(D,32)
            #build X and y
            X = []
            y = []
            episode_mse = []
            for episode in sample:
                t_s,t_a,t_reward,t_s_prime,terminal = episode

                #calculate bellman update equation
                delta_q = t_reward
                if not terminal:
                    s_prime_f = np.array([t_s_prime])
                    delta_q = t_reward + gamma*max(target_model.predict(s_prime_f)[0]) #TODO: change to target model later

                #update Q(s,a)
                s_f = np.array([t_s])
                q_values = model.predict(s_f)
                q_values[0][t_a] = delta_q
                #update neural network with new target
                X.append(t_s)
                y.append(q_values[0])

            X = np.array(X)
            y = np.array(y)

            history = model.fit(X,y,epochs = 1, verbose=0)
            #----------------------------------------------------------------------------------------------------------------
            mse_arr.extend(history.history["mean_squared_error"])
            moving_avg = (total_reward - avg_reward_arr[-1]) * (2/(len(avg_reward_arr) +1)) + avg_reward_arr[-1]
            avg_reward_arr.append(moving_avg)
            #transfer weights to target Q
            c+=1
            #TODO: idk after how many steps should the weights be transfered
            if c == 16:
                #print ("tranfered weights...")
                target_model.set_weights(model.get_weights())
                c = 0
        reward_arr.append(total_reward)

        # plt.figure(2)
        # plt.plot(avg_reward_arr, color="g")
        # plt.title("current avg reward")
        # plt.pause(0.05)

        # plt.figure(1)
        # plt.plot(reward_arr, color="b")
        # plt.pause(0.05)

        # plt.figure(0)
        # plt.plot(mse_arr, color="r")
        # plt.pause(0.05)

    plt.show()
    env.close()
    #---save model---
    # serialize model to JSON
    model_json = target_model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    target_model.save_weights("model.h5")
    print("Saved model to disk")

Q_learn()
