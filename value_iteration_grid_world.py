import numpy as np

states = [[0,0,0],[0,0,0],[0,0,0]]
values = states.copy()
R = [[-1,-1,-1],[-1,10,-1],[-1,-1,-1]]
#stored as translation to x,y: up, down, left, right
actions = [[-1,0],[1,0],[0,-1],[0,1]]
#probability of actually going up,down,left,right
probability = {(-1,0): 0.25,(1,0):0.25, (0,-1):0.25,(0,1):0.25}

def value_iteration():
    gamma = 0.9

    max_iters = 1000 #number of iterations before convergence
    #compute value function for each state
    for x in range (max_iters):
        for i in range(len(states)):
            for j in range (len(states)):
                s = states[i][j]
                res = []
                #one step look ahead
                for a in actions:
                    #get next state
                    try:
                        s_prime = values[i+a[0]][j+a[1]]
                        r = R[i][j]
                        Q = probability.get(tuple(a)) * (r + gamma*s_prime)
                        #Q = probability.get(tuple(a)) * (r + gamma*s_prime)
                        res.append(Q)
                    except IndexError:
                        continue
                #store optimal value given an action
                #IM NOT SURE IF THE VALUE IS THE MAXIMUM OR THR SUM OF POSSIBLE VALUES!!
                values[i][j] = max(res)
    return values

print (value_iteration())
