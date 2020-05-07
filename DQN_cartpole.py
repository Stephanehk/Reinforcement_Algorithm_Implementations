import gym
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense,Activation,BatchNormalization
from keras.optimizers import SGD
from keras.models import model_from_json
import matplotlib.pyplot as plt
import pprint

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
                reservoir[j] = arr[i]
    return reservoir

def get_best_action (model, s):
    s = np.array(s)
    s = s.reshape(1,4)
    action_values = model.predict(s)
    return np.argmax(action_values[0])


def epsilon_greedy(a):
    if random.uniform(0,1) < (1-eps):
        return a
    else:
        return random.choice(possible_actions)

def Q_learn ():
    env = gym.make("CartPole-v0")

    #memory replay array
    D = []

    num_episodes = 200
    c = 0

    def creat_model():
        opt = SGD(lr=0.01)
        #try out different configs of batch norm
        model = Sequential()
        model.add(Dense(units=12, input_dim=4, activation="relu"))
        model.add(Dense(units=8, activation="relu"))
        model.add(Dense(units=2, activation='linear'))
        model.compile(loss='mean_squared_error',optimizer=opt, metrics = ["mse","accuracy"])
        return model

    #setup neural network
    model = creat_model()
    #setup target neural network
    target_model = creat_model()


    reward_arr = []
    mse_arr = []
    acc_arr = []
    for i in range (num_episodes):
        s = env.reset()
        done = False
        total_reward = 0
        t = 0

        while not done:
            env.render()
            t+=1

            #memory replay - sample 64 (s,a,r,s_prime) pairs to train NN on
            if len (D)> 64:
                a = epsilon_greedy(get_best_action (model, s))
                #run single instance of episode
                s_prime, reward, done, info = env.step(a)
                total_reward += reward
                D.append((s,a,reward,s_prime,done))


                sample =reservoir_sample(D,64)
                #build X and y
                X = []
                y = []
                episode_mse = []
                for episode in sample:
                    s,a,reward,s_prime,terminal = episode

                    #calculate bellman update equation
                    delta_q = reward
                    if not terminal:
                        s_prime_f = np.array([s_prime])
                        delta_q = reward + gamma*max(target_model.predict(s_prime_f)[0])

                    #update Q(s,a)
                    s_f = np.array([s])
                    q_values = model.predict(s_f)
                    q_values[0][a] = delta_q
                    #update neural network with new target
                    X.append(s)
                    y.append(q_values[0])

                X = np.array(X)
                y = np.array(y)

                history = model.fit(X,y,epochs = 1, verbose=0)
                mse_arr.extend(history.history["mean_squared_error"])
                acc_arr.extend(history.history["acc"])

            else:
                a = epsilon_greedy(random.choice(possible_actions))
                #run single instance of episode
                s_prime, reward, done, info = env.step(a)
                total_reward += reward
                D.append((s,a,reward,s_prime,done))


            s = s_prime
            #transfer weights to target Q
            c+=1
            #TODO: idk after how many steps should the weights be transfered
            if c == 16:
                #print ("tranfered weights...")
                color = "b"
                target_model.set_weights(model.get_weights())
                c = 0
        reward_arr.append(total_reward)

        plt.figure(2)
        plt.plot(acc_arr, color="g")
        plt.pause(0.05)

        plt.figure(1)
        plt.plot(reward_arr, color="k")
        plt.pause(0.05)

        plt.figure(0)
        plt.plot(mse_arr, color="r")
        plt.pause(0.05)

        if done:
            print("Episode " + str(i) + " finished after {} timesteps".format(t+1))

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
