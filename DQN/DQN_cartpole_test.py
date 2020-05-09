import gym
import numpy as np
import random
from keras.optimizers import SGD
from keras.models import model_from_json
import matplotlib.pyplot as plt
import timeit

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

    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("model.h5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    opt = SGD(lr=0.01)
    model.compile(loss='mean_squared_error',optimizer=opt, metrics = ["mse"])


    num_episodes = 1000

    total_reward_arr = []

    for i in range (num_episodes):
        s = env.reset()
        done = False
        total_reward = 0
        t = 0

        while not done:
            env.render()
            t+=1
            a = epsilon_greedy(get_best_action (model, s))
            #run single instance of episode
            s_prime, reward, done, info = env.step(a)
            total_reward += reward
            s = s_prime

        if done:
            total_reward_arr.append(total_reward)
            #print("Episode " + str(i) + " finished after {} timesteps".format(t+1))
            if i%100:
                print ("Avg reward over last " + str(i) + " iterations: " + str(np.mean(total_reward_arr)))
                total_reward_arr.clear()
    env.close()

Q_learn()
