# import
import math
import random
import sys
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import PIL.Image
from Q_network import *
from basic_tool import *
from sklearn.neighbors import NearestNeighbors

import torch
import torch.nn as nn
import torch.nn.functional as F

from IPython.display import clear_output

random.seed

# Reload
## DQN's hyper para
n_actions = 5
# state: RSSI (520), coordinate (2), radius (1), history (50)
n_states = 520 + 2 + 1 + 50 
n_hidden = 512
batch_size = 100
gamma = 0.1 # reward discount factor
target_replace_iter = 100
memory_capacity = 200
n_episodes = 1000
lr = 0.01
eps = 0
max_search_steps = 10
it = 0
cost_it = 0
avg_it = 0
delta = 0.7
log_step = 500
Rewards = 0
radius_gt = 0.5
each_search_it = np.zeros((n_episodes,10))
it_list = []
total_distance = 0
total_IoW = 0
dis_list = []
IoW_list = []

dqn = DQN(n_states, n_actions, n_hidden, batch_size, lr, gamma, target_replace_iter, memory_capacity)
# dqn.load("IPS_dqn.pt")

color = ['#F5B4B3', '#CA885B', '#DAE358', '#9DE358', '#58E39D', '#58E3E1', '#58A2E3', '#5867E3', '#9D58E3', '#E158E3', '#E358B0', '#E35869']

# Train_data
floor  = 2
building = 0
wifi_loc_time = dataloader(floor, building)

## Map boundaries
longitude_list = np.array([max(wifi_loc_time[:, 520]), -1\
                            , min(wifi_loc_time[:, 520])])
latitude_list = np.array([max(wifi_loc_time[:, 521]), -1\
                            , min(wifi_loc_time[:, 521])])

## KNN initial calculation
nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(wifi_loc_time[:,:520])
distances, indices = nbrs.kneighbors(wifi_loc_time[:,:520])

## DQN training
for k in range(n_episodes):
    print("Epoch - ", k, "eps : ", eps)
    avg_it = 0
    total_distance = 0
    total_IoW = 0
    for i in range(len(wifi_loc_time)):
        # some important variables used for training
        Rewards = 0
        Goal = False
        alpha = 0.7
        next_coordinate = np.array([0, 0])
        next_radius = 0
        rect1 = []
        
        ## 1. KNN locates initial coordinates and radius
        Hx = 0.
        Hy = 0.
        for m in range(3):
            Hx += wifi_loc_time[indices[i, m], 520]
            Hy += wifi_loc_time[indices[i, m], 521]
        Hx /= 3.
        Hy /= 3.
        coordinate = np.array([Hx, Hy])
        radius = 1.
        # initial history, 5n vector
        history = np.zeros(shape=(5*max_search_steps,), dtype=int)
        
        ## 2. Check initial KNN IoW
        while True:
            IoW_cur = IoW(wifi_loc_time[i, 520:522], coordinate, radius_gt, radius)
            if (IoW_cur == 0):
                radius *= 1.5
            elif (IoW_cur > delta):
#                         print("Precise location!")
                Goal = True
                break
            else:
                break
        if (Goal == True):
            continue
        # initial state: RSSI (520), coordinate (2), radius (1), history (50)
        state = np.concatenate((wifi_loc_time[i, :520], coordinate.copy(), np.array([radius]), history.copy()), axis=0)
        
        ## Plot gt region
        if k % log_step +300 == 0:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.xlim(longitude_list[0], longitude_list[2])
            plt.ylim(latitude_list[0], latitude_list[2])
            rect0 = plt.Rectangle((wifi_loc_time[i,520]-radius_gt, wifi_loc_time[i,521]-radius_gt), 2*radius_gt, 2*radius_gt, alpha=0.9)
            rect1.append(plt.Rectangle((coordinate[0]-radius, coordinate[1]-radius), 2*radius, 2*radius, alpha = 0.6, color = color[-1]))
        
        ## 3. Searching starts
        for t in range(max_search_steps):
            right = right_action(wifi_loc_time[i, 520:522], coordinate, radius_gt, radius)
            it = 0
            cost_it = 0
            if (radius < 0.5):
                # print("Radius is small enough and searching ends")
                Goal = True
                break
            
            
            while True:
                it += 1
                ### (1) select an action
                action = dqn.choose_action(state, eps, i, t, it)
                # print("Loc", i, "round ", t, "-", it, "times, [", action, "], expected", right)
                ### (1) - 1. New Center
                ### 0 -> "Up Left"
                ### 1 -> "Up Right"
                ### 2 -> "Down Left"
                ### 3 -> "Down Right"
                ### 4 -> "Center"
                next_coordinate = coordinate.copy()
                if (action == 0):
                    next_coordinate[0] -= radius/2.
                    next_coordinate[1] += radius/2.
                elif (action == 1):
                    next_coordinate[0] += radius/2.
                    next_coordinate[1] -= radius/2.
                elif (action == 2):
                    next_coordinate[0] -= radius/2.
                    next_coordinate[1] -= radius/2.
                elif (action == 3):
                    next_coordinate[0] += radius/2.
                    next_coordinate[1] += radius/2.
                else:
                    next_coordinate = coordinate
                ### (1) - 2. New radius
                next_radius = radius * alpha
                ### (1) - 3. New IoW
                next_IoW = IoW(wifi_loc_time[i, 520:522], next_coordinate, radius_gt, next_radius)
                # print("  IoW", IoW_cur, "->", next_IoW)
                if (next_IoW > delta and action == right):
                    print("Precise location!")
                    Goal = True
                    cost_it += it
                    del next_coordinate
                    break
                elif (next_IoW > IoW_cur and action == right):
                    # close score
                    reward = distance_progress(wifi_loc_time[i, 520:522], coordinate, next_coordinate)
                    # print("  [C]  ontinue next round of searching, reward", reward, "\n   location", coordinate, "->", next_coordinate, "\n   radius", radius,"->" , next_radius)
                    IoW_cur = next_IoW
                    next_history = history.copy()
                    one_hot = t*5 + action
                    next_history[one_hot] = 1
                    next_state = np.concatenate((wifi_loc_time[i,:520], next_coordinate, np.array([next_radius]), next_history), axis=0)
#                             dqn.store_transition(state.copy(), action, reward, next_state.copy())
                    radius = next_radius
                    coordinate = next_coordinate
                    history = next_history.copy()
                    state = next_state.copy()
                    Rewards += reward
                    del next_history
                    del next_state
                    del next_coordinate
                    cost_it += it
                    if k % log_step + 300 == 0:
                        # Plot
                        rect1.append(plt.Rectangle((coordinate[0]-radius, coordinate[1]-radius), 2*radius, 2*radius, alpha = 0.6, color = color[t]))
                    break
                else:
                    reward = distance_progress(wifi_loc_time[i, 520:522], coordinate, next_coordinate)
                    next_history = history.copy()
                    one_hot = t*5 + action
                    next_history[one_hot] = 1
                    next_state = np.concatenate((wifi_loc_time[i,:520], next_coordinate, np.array([next_radius]), next_history), axis=0)
                    Rewards += reward
                    # back_propagation
                    dqn.BP(state, next_state, reward, right, i, t, it)
                    # print("  [R]  epeat this round, reward", reward, "\n   location", coordinate, "->", next_coordinate, "\n   radius", radius,"->" , next_radius)
                if it > 9:
                    cost_it += it
                    # print("--------------------------Training fail!---------------------------")
                    break
            
            each_search_it[k][t] += cost_it
            
            if (Goal == True):
                print("End searching for ", i ,". Round", t,"costs ", cost_it)
                break
        clear_output(wait=True)
        # distance
        total_distance += dis(wifi_loc_time[i, 520:522], coordinate)
        total_IoW += IoW_cur
        del coordinate
        del state
        if k % log_step + 300 == 0:
            for x in rect1:
                ax.add_patch(x)
            ax.add_patch(rect0)
            plt.title("loc: "+str(i)+" epoch "+str(k)+" cost total searching "+str(cost_it)+" times")
            plt.savefig("loc_"+str(i)+"_e_"+str(k))
            plt.close()
            plt.cla()
            plt.clf()
            rect1.clear()
#                 if dqn.memory_counter > memory_capacity:
#                     dqn.learn()
        avg_it += cost_it
    
    it_list.append(avg_it/len(wifi_loc_time))
    total_distance /= float(len(wifi_loc_time))
    dis_list.append(total_distance)
    total_IoW /= float(len(wifi_loc_time))
    IoW_list.append(total_IoW)
    each_search_it /= float(len(wifi_loc_time))
    print("Epoch {} - avg dis : {}, avg IoW : {} ; ".format(k, total_distance, total_IoW), each_search_it)
    torch.save(dqn.eval_net.state_dict(), "model_{}.pt".format(k))

plt.plot([x for x in range(len(dis_list))], dis_list)
plt.plot([x for x in range(len(dis_list))], IoW_list)
plt.savefig("Trend_of_dis_IoW.png")
for i in range(10):
    plt.plot([x for x in range(len(dis_list))], each_search_it[:,i], color=color[i])
    plt.savefig("Trend_of_search_it.png")


# Test_data
floor  = 2
building = 0
wifi_loc_time = dataloader(floor, building, filepath="1478167721_0345678_validationData.csv")

## Map boundaries
longitude_list = np.array([max(wifi_loc_time[:, 520]), -1\
                            , min(wifi_loc_time[:, 520])])
latitude_list = np.array([max(wifi_loc_time[:, 521]), -1\
                            , min(wifi_loc_time[:, 521])])

## KNN initial calculation
distances, indices = nbrs.kneighbors(wifi_loc_time[:,:520])

## DQN training

avg_it = 0
total_distance = 0
total_IoW = 0
for i in range(len(wifi_loc_time)):
    # some important variables used for training
    Rewards = 0
    Goal = False
    alpha = 0.7
    next_coordinate = np.array([0, 0])
    next_radius = 0
    rect1 = []
    
    ## 1. KNN locates initial coordinates and radius
    Hx = 0.
    Hy = 0.
    for m in range(3):
        Hx += wifi_loc_time[indices[i, m], 520]
        Hy += wifi_loc_time[indices[i, m], 521]
    Hx /= 3.
    Hy /= 3.
    coordinate = np.array([Hx, Hy])
    radius = 1.
    # initial history, 5n vector
    history = np.zeros(shape=(5*max_search_steps,), dtype=int)
    
    ## 2. Check initial KNN IoW
    while True:
        IoW_cur = IoW(wifi_loc_time[i, 520:522], coordinate, radius_gt, radius)
        if (IoW_cur == 0):
            radius *= 1.5
        elif (IoW_cur > delta):
#                         print("Precise location!")
            Goal = True
            break
        else:
            break
    if (Goal == True):
        continue
    # initial state: RSSI (520), coordinate (2), radius (1), history (50)
    state = np.concatenate((wifi_loc_time[i, :520], coordinate.copy(), np.array([radius]), history.copy()), axis=0)
    
    ## Plot gt region
    if k % log_step == 0:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.xlim(longitude_list[0], longitude_list[2])
        plt.ylim(latitude_list[0], latitude_list[2])
        rect0 = plt.Rectangle((wifi_loc_time[i,520]-radius_gt, wifi_loc_time[i,521]-radius_gt), 2*radius_gt, 2*radius_gt, alpha=0.9)
        rect1.append(plt.Rectangle((coordinate[0]-radius, coordinate[1]-radius), 2*radius, 2*radius, alpha = 0.6, color = color[-1]))
    
    ## 3. Searching starts
    for t in range(max_search_steps):
        right = right_action(wifi_loc_time[i, 520:522], coordinate, radius_gt, radius)
        it = 0
        cost_it = 0
        if (radius < 0.5):
            # print("Radius is small enough and searching ends")
            Goal = True
            break
        
        
        while True:
            it += 1
            ### (1) select an action
            action = dqn.choose_action(state, eps, i, t, it)
            # print("Loc", i, "round ", t, "-", it, "times, [", action, "], expected", right)
            ### (1) - 1. New Center
            ### 0 -> "Up Left"
            ### 1 -> "Up Right"
            ### 2 -> "Down Left"
            ### 3 -> "Down Right"
            ### 4 -> "Center"
            next_coordinate = coordinate.copy()
            if (action == 0):
                next_coordinate[0] -= radius/2.
                next_coordinate[1] += radius/2.
            elif (action == 1):
                next_coordinate[0] += radius/2.
                next_coordinate[1] -= radius/2.
            elif (action == 2):
                next_coordinate[0] -= radius/2.
                next_coordinate[1] -= radius/2.
            elif (action == 3):
                next_coordinate[0] += radius/2.
                next_coordinate[1] += radius/2.
            else:
                next_coordinate = coordinate
            ### (1) - 2. New radius
            next_radius = radius * alpha
            ### (1) - 3. New IoW
            next_IoW = IoW(wifi_loc_time[i, 520:522], next_coordinate, radius_gt, next_radius)
            # print("  IoW", IoW_cur, "->", next_IoW)
            if (next_IoW > delta):
                print("Precise location!")
                coordinate = next_coordinate
                radius = next_radius
                Goal = True
                cost_it += it
                del next_coordinate
                break
            else:
                # close score
                reward = distance_progress(wifi_loc_time[i, 520:522], coordinate, next_coordinate)
                # print("  [C]  ontinue next round of searching, reward", reward, "\n   location", coordinate, "->", next_coordinate, "\n   radius", radius,"->" , next_radius)
                IoW_cur = next_IoW
                next_history = history.copy()
                one_hot = t*5 + action
                next_history[one_hot] = 1
                next_state = np.concatenate((wifi_loc_time[i,:520], next_coordinate, np.array([next_radius]), next_history), axis=0)
#                             dqn.store_transition(state.copy(), action, reward, next_state.copy())
                radius = next_radius
                coordinate = next_coordinate
                history = next_history.copy()
                state = next_state.copy()
                Rewards += reward
                del next_history
                del next_state
                del next_coordinate
                cost_it += it
                if k % log_step == 0:
                    # Plot
                    rect1.append(plt.Rectangle((coordinate[0]-radius, coordinate[1]-radius), 2*radius, 2*radius, alpha = 0.6, color = color[t]))
                break
            
        
        each_search_it[t] += cost_it
        
        if (Goal == True):
            print("End searching for ", i ,", costs ", cost_it)
            break
    clear_output(wait=True)
    # distance
    total_distance += dis(wifi_loc_time[i, 520:522], coordinate)
    total_IoW += IoW_cur
    del coordinate
    del state
    if k % log_step == 0:
        for x in rect1:
            ax.add_patch(x)
        ax.add_patch(rect0)
        plt.title("loc: "+str(i)+" epoch "+str(k)+" cost total searching "+str(cost_it)+" times")
        plt.savefig("loc_"+str(i)+"_e_"+str(k))
        plt.close()
        plt.cla()
        plt.clf()
        rect1.clear()
#                 if dqn.memory_counter > memory_capacity:
#                     dqn.learn()
    avg_it += cost_it

it_list.append(avg_it/len(wifi_loc_time))
total_distance /= float(len(wifi_loc_time))
total_IoW /= float(len(wifi_loc_time))
each_search_it /= float(len(wifi_loc_time))
print("avg dis : {}, avg IoW : {} ; ".format(total_distance, total_IoW), each_search_it)

torch.save(dqn.state_dict(), "model.pt")