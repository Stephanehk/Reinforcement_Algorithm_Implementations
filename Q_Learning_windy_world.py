import numpy as np
import random

states = [  [0,1,1,2,2,0],
            [0,1,1,2,2,0],
            [0,1,1,2,2,0],
            [0,1,1,2,"E",0],
            [0,1,1,2,2,0],
            [0,1,1,2,2,0]]
possible_actions = [(-1,0),(1,0),(0,-1),(0,1)]
gamma = 0.9
eps = 0.1

def generate_path(s,policy):
    path = [s]
    x,y = s
    s_val = states[x][y]

    while s_val != "E":
        #print (s)
        a_x,a_y = policy.get(s)
        s = (x+a_x, y+a_y - states[x+a_x][y+a_y])
        x,y = s
        s_val = states[x][y]
        print (s_val)
        path.append(s)
    return path

def epsilon_greedy(Q, s_prime):
    if random.uniform(0,1) < (1-eps):
        #-------------THIS GETS NEXT ACTION FROM MAX Q VAL -------------------------
        max_q_val = max([Q.get((s_prime,a)) for a in possible_actions])
        for s_a, q_val in Q.items():
            dict_s,dict_a = s_a
            if q_val == max_q_val and dict_s == s_prime:
                return dict_a
    else:
        return random.choice(possible_actions)

def generate_full_episode (policy):
    #initialize parameters
    current_x, current_y = 0, 3
    s = states[current_x][current_y]
    n_steps = 0

    s_a_pairs = {}
    s_r_pairs = {}

    #execute policy
    while s != "E" and n_steps < 100:
        n_steps+=1
        x,y = policy.get((current_x, current_y))
        #TODO: redundent dictionary - figure out a better implementation to optimize this out
        s_a_pairs[(current_x, current_y)] = (x,y)
        #make sure everything is in bounds
        if len(states[0]) > current_x + x > -1:
            current_x = current_x + x
        if len(states) > current_y + y > -1:
            if len(states) > current_y + y + states[current_x][current_y + y] > -1:
                current_y = current_y + y + states[current_x][current_y + y]
            else:
                current_y = current_y + y
        print (str(current_x) + "," + str(current_y))
        s = states[current_x][current_y]
    #determine rewards
    if s == "E":
        out = 1
    else:
        out = -1

    visited_states = list(s_a_pairs.keys())
    for i in range (len(visited_states)):
        s = visited_states[len(visited_states)-i-1]
        s_r_pairs[s] = np.power(out,i+1) * np.power(gamma,i+1)

    return s_a_pairs, s_r_pairs

def generate_episode_step (state, action):
    #initialize parameters
    current_y, current_x = state
    y,x = action

    #make sure everything is in bounds
    if len(states[0]) > current_x + x > -1:
        current_x = current_x + x

    if len(states) > current_y + y > -1:
        if type (states[current_y+y][current_x]) == str:
            current_y = current_y + y
        else:
            current_y = current_y + y - states[current_y + y][current_x]
            #put current_y back in bounds if it is blown out
            if current_y < 0:
                current_y = 0
            elif current_y > len(states)-1:
                current_y = len(states)-1

    if states[current_y][current_x] == "E":
        return (current_y, current_x), 1
    #state, action
    return (current_y, current_x), -1

def Q_learn ():
    #generate random policy
    policy = {}
    #store value for each state action pair
    Q = {}
    for i in range (len(states)):
        for j in range (len(states[0])):
            policy[(i,j)] = 0
            for a in possible_actions:
                Q[((i,j),a)] = 0

    num_episodes = 8000
    alpha = 0.5

    for i in range (num_episodes):
        s = (3,0)
        s_val = "s"
        while s_val!= "E":
            a = epsilon_greedy(Q, s)
            #run single instance of episode
            s_prime, R = generate_episode_step(s, a)

            #get next best action
            max_q_val = max([Q.get((s_prime,a)) for a in possible_actions])
            for s_a, q_val in Q.items():
                dict_s,dict_a = s_a
                if q_val == max_q_val and dict_s == s_prime:
                    a_prime = dict_a
                    break

            #calculate Q
            Q[(s,a)] = Q.get((s,a)) + alpha*(R + gamma*Q.get((s_prime,a_prime)) - Q.get((s,a)))

            s = s_prime

            #update policy (only used to check the outcome)
            policy[s] = a_prime

            #get next state value - only used for termination if end state has been reached
            x,y = s
            s_val = states[x][y]
    return policy

policy = Q_learn()
print (policy)
