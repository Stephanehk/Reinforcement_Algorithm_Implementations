import numpy as np
import random

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

cards = [2,3,4,5,6,7,8,9,10,10,10,11]

def plot_policy (policy):
    #(sum, dealers_card)
    X = []
    Y = []
    Z = []
    #format everything
    for val, key in zip(list(policy.keys()), list(policy.values())):
        sum, dealers_card = val
        X.append(sum)
        Y.append(dealers_card)
        Z.append(key)
    print (Y)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(X, Y, Z)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z');
    plt.show()

def percentage_wins(policy):
    wins = 0
    for i in range (1000):
        seen_card = random.choice(cards)
        dealers_hand = seen_card + random.choice(cards)
        hand = random.choice(cards)+ random.choice(cards)

        while policy[(hand,seen_card)] == 1:
            hand += random.choice(cards)
            if hand >= 21:
                break
        if hand == 21 or 21 > hand > dealers_hand:
            wins+=1
    return (wins/1000)

def play_game (policy,gamma):
    out = None
    s_a_pairs = {}
    s_r_pairs = {}

    seen_card = random.choice(cards)
    dealers_hand = seen_card + random.choice(cards)
    hand = random.choice(cards) + random.choice(cards)
    #store state action pair
    s_a_pairs[(hand,seen_card)] = policy[(hand,seen_card)]

    action = epsilon_greedy(policy[(hand,seen_card)])
    while action == 1:
        c = random.choice(cards)
        hand += c

        if hand >= 21:
            break
        else:
            s_a_pairs[(hand,seen_card)] = policy[(hand,seen_card)] #ignore the last state
        action = epsilon_greedy(policy[(hand,seen_card)])

    if hand > 21 or hand < dealers_hand:
        out = -1
    elif hand > dealers_hand:
        out = 1
    else:
        out = 0

    states = list(s_a_pairs.keys())
    for i in range (len(states)):
        s = states[len(states)-i-1]
        s_r_pairs[s] = np.power(out,i+1) * np.power(gamma,i+1)

    return out,s_a_pairs, s_r_pairs

def epsilon_greedy(a, eps=0.1):
    if random.uniform(0,1) < (1-eps):
        return a
    else:
        return random.choice([0,1])

def monte_carlo ():
    gamma = 1
    #initialize random policy
    policy = {}
    #initialize returns dictionary for every state action pair
    Q = {}
    #initialize counter dictionary for every state action pair
    N = {}
    for c1 in cards:
        for c2 in cards:
            for c3 in cards:
                policy[(c1+c2,c3)] = random.choice([0,1])

                Q[((c1+c2,c3), 0)] = 0
                Q[((c1+c2,c3), 1)] = 0

                N[((c1+c2,c3), 0)] = 0
                N[((c1+c2,c3), 1)] = 0

    #maximum number of iterations until convergence
    max_iters = 100000

    #stuff for plotting
    prev_q = None
    deltas = []

    for i in range (max_iters):
        if (i%100000) == 0:
            print ("On episode " + str(i))

        r, s_a_pairs, s_r_pairs = play_game (policy,gamma)

        biggest_change = 0
        for s,a in zip(list(s_a_pairs.keys()), list(s_a_pairs.values())):
            #print ("looking at " + str(s) + " - " + str(a))

            #print (str(s) + " - " + str(a))
            N[(s,a)] = N.get((s,a)) + 1

            #incremental mean formula
            alpha = (1/N.get((s,a)))
            #alpha = 0.8
            prev_q = Q.get((s,a))

            Q[(s,a)] = Q.get((s,a)) + alpha*(s_r_pairs.get(s) - Q.get((s,a)))
            #OR
            #Q[s][a] = np.mean(returns[s][a])
            biggest_change = max(biggest_change, np.abs(Q.get((s,a)) - prev_q))
        deltas.append(biggest_change)

        for s in policy.keys():
            if Q.get((s,0)) > Q.get((s,1)):
                policy[s] = 0
            else:
                policy[s] = 1
    # plt.plot(deltas)
    # plt.show()
    return policy

opimal_policy = monte_carlo ()
print (opimal_policy)
print ("\n")
print(percentage_wins(opimal_policy))
plot_policy (opimal_policy)
