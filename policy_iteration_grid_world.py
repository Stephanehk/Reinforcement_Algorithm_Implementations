import numpy as np
import random

states = [[0,0,0],
          [0,0,0],
          [0,0,0]]
R = [[-1,-1,-1],[-1,10,-1],[-1,-1,-1]]
#stored as translation to x,y: nothing, up, down, left, right
actions = [[0,0],[-1,0],[1,0],[0,-1],[0,1]]
#probability of actually going up,down,left,right
probability = {(0,0):0.2,(-1,0): 0.2,(1,0):0.2, (0,-1):0.2,(0,1):0.2}
gamma = 0.9

def evaluate_policy(policy,V):
    V_prime = V.copy()
    for i in range(len(states)):
        for j in range (len(states[0])):
            #store value given the policy at current state
            #try catch statement is only here for the initial policy
            try:
                a = actions[policy[i][j]]
                V_prime[i][j] = probability.get(tuple(a))*(R[i+a[0]][j+a[1]] + gamma*V[i+a[0]][j+a[1]]) + (1-probability.get(tuple(a)))*(R[i][j]+gamma*V[i][j])
            except IndexError:
                continue
    return V_prime

def improve_policy(policy, V):
    for i in range(len(states)):
        for j in range (len(states[0])):
            s = states[i][j]
            res = []
            #one state look ahead
            for a in actions:
                #get next state
                try:
                    Q = probability.get(tuple(a)) * (R[i+a[0]][j+a[1]] + gamma*V[i+a[0]][j+a[1]])+ (1-probability.get(tuple(a)))*(R[i][j]+gamma*V[i][j])
                    res.append(Q)
                except IndexError:
                    continue
            #store the optimal action to take at current state
            policy[i][j] = res.index(max(res))
    return policy

def policy_iteration():
    V = states.copy()
    #initialize policy with random actions
    policy = [[random.randint(0,3) for i in range (len(states))] for j in range (len(states[0]))]
    #compute value function for each state
    k = 0
    while k < 100:
        V = evaluate_policy(policy,V)
        new_policy = improve_policy(policy, V)
        if new_policy == policy:
            k +=1
        else:
            policy = new_policy
            k=0
    return policy

print (policy_iteration())
