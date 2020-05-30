import numpy as np
import gym
import random
import matplotlib.pyplot as plt
import warnings
from keras.models import Sequential
from keras.layers import Dense,Activation,BatchNormalization, Input
from keras.optimizers import SGD,Adam
from keras.models import model_from_json, Model
from keras import backend as K

#warnings.filterwarnings('error')

env = gym.make("LunarLander-v2")
env = gym.make("CartPole-v0")
n_features = 4
n_actions = env.action_space.n

def follow_policy(policy,s, return_action):
    probs = policy.predict(np.array([s]))[0]
    if return_action:
        return np.random.choice(n_actions,p=probs)
    else:
        return probs

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

def encode_a(a):
    a_encoded = np.zeros(n_actions)
    a_encoded[a] = 1
    return a_encoded

def format_sa_pair (s,a):
    x = list(s.copy())
    a_encoded = encode_a(a)
    x.extend(a_encoded)
    return x

def phi (s,a):
    phi = np.zeros([n_actions,n_features])
    phi[a] = s
    #phi = phi.flatten()
    return phi

def model(input_dim_, output_dim_, alpha, model_type):

    opt = Adam(lr = alpha)
    input = Input(shape=(input_dim_,))
    delta = Input(shape=[1])
    l1 = Dense(units = 128, input_dim = input_dim_, activation = "relu")(input)
    l2 = Dense(units=128, activation="relu")(l1)

    def loss_function (y,y_pred):
        y_pred = K.clip(y_pred,1e-8,1-1e-8)
        #select action taken
        # log_lik = K.sum(y*K.log(y_pred))
        # return K.mean(-log_lik*delta)
        log_lik = y*K.log(y_pred)
        return K.sum(-log_lik*delta)

    if model_type == "actor":
        out = Dense(units = output_dim_, activation="softmax")(l2)
        model = Model(input=[input,delta], output = [out])
        model.compile(loss = loss_function,optimizer=opt)
    elif model_type == "critic":
        out = Dense(units = output_dim_, activation="linear")(l2)
        model = Model(input=[input], output=[out])
        model.compile(loss = "mean_squared_error",optimizer=opt, metrics=["mse"])
    elif model_type == "policy":
        out = Dense(units = output_dim_, activation="softmax")(l2)
        model = Model(input=[input], output = [out])
    else:
        print ("cannot understand model type")
        return None
    return model

def actor_critic():
    #init everything
    gamma = 0.99
    alpha_theta = 0.01
    alpha_w = 0.01
    num_iters = 2000
    memory_replay = []

    #init weights
    # thetas = np.array([[random.uniform(0, 1) for i in range(n_actions)] for j in range (n_features)])
    # W = np.array([[random.uniform(0, 1) for i in range(n_features)] for j in range (n_actions)])

    #predicts action given s
    policy = model(n_features,n_actions,alpha_theta,"policy")
    #used only for training
    actor = model(n_features,n_actions,alpha_theta,"actor")
    #predicst future reward given (s,a) pair
    critic = model(n_features + n_actions,1,alpha_w,"critic")


    avg_reward_arr = [0]
    avg_timestep_arr = [0]
    all_rewards = []

    actor_loss = []
    critic_loss = []

    #debugging stuff
    first_episode_reward = None
    for i in range(num_iters):
        done = False
        s = env.reset()
        t = 0
        total_reward = 0
        discounted_i = 1
        while not done:
            #env.render()
            t+=1
            a = follow_policy(policy,s, True)
            s_prime,r, done, _ = env.step(a)
            total_reward+=r

            if not done:
                a_prime = follow_policy(policy,s_prime, True)
                #format critic X
                r += gamma*critic.predict(np.array([format_sa_pair (s_prime,a_prime)]))[0]
                #r_t += gamma*critic.predict([s_prime_t,a_prime])[0] - critic.predict([s_t,a_t])[0]

            # #----------------ACTOR---------------------------------------
            #compute delta
            delta = r - critic.predict(np.array([format_sa_pair (s,a)]))
            #print (len(delta))
            #update actor to predict best action given state
            s_reshaped = s.reshape(1,len(s))
            actor_X = [np.array(s_reshaped),np.array(delta)]
            actor_y = np.array([encode_a(a)])
            #X = np.array(X)
            actor_history = actor.fit(actor_X,actor_y,epochs = 1, verbose=0)
            actor_loss.append(actor_history.history["loss"])
            #----------------CRITIC---------------------------------------
            #format critic x
            x = format_sa_pair (s,a)
            x = np.array([x])
            #update critic to predict the future reward after one step
            critic_history = critic.fit(x,[r],epochs = 1, verbose=0)
            critic_loss.append(critic_history.history["mean_squared_error"])

            s = s_prime

            if done:
                if i == 0:
                    first_episode_reward = total_reward
                t+=1
                print("Episode finished after {} timesteps".format(t))
                moving_avg = (total_reward - avg_reward_arr[-1]) * (2/(len(avg_reward_arr) +1)) + avg_reward_arr[-1]
                avg_reward_arr.append(moving_avg)

                moving_avg = (t - avg_timestep_arr[-1]) * (2/(len(avg_timestep_arr) +1)) + avg_timestep_arr[-1]
                avg_timestep_arr.append(moving_avg)

                all_rewards.append(t)
                env.close()
                break

        # plt.figure(0)
        # plt.plot(avg_reward_arr, color = "k")
        # plt.pause(0.05)

        plt.figure(1)
        plt.plot(actor_loss, color = "g")
        plt.pause(0.05)

        plt.figure(2)
        plt.plot(critic_loss, color = "b")
        plt.pause(0.05)


        plt.figure(3)
        plt.plot(all_rewards, color = "r")
        plt.pause(0.05)
    plt.show()

actor_critic()
