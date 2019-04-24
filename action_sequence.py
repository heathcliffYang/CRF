 # import
import math
import random
import sys
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import PIL.Image
from rnn import *
from basic_tool import *
from sklearn.neighbors import NearestNeighbors

import torch
import torch.nn as nn
import torch.nn.functional as F

# Setting
np.set_printoptions(precision=3)
random.seed

# Load data
floor  = 2
building = 0
wifi_loc_time = dataloader(floor, building)
wifi_loc_time_test = dataloader(floor, building, filepath="1478167721_0345678_validationData.csv")

## Map boundaries
longitude_list = np.array([max(wifi_loc_time[:, 520]), -1\
                            , min(wifi_loc_time[:, 520])])
latitude_list = np.array([max(wifi_loc_time[:, 521]), -1\
                            , min(wifi_loc_time[:, 521])])

## KNN initial calculation
nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(wifi_loc_time[:,:520])
distances, indices = nbrs.kneighbors(wifi_loc_time[:,:520])

# Training setting
n_episodes = 300
total_loss = np.zeros((n_episodes,))
radius_gt = 0.5
delta = 0.5
avg_distance = np.zeros((n_episodes,2))
avg_iow = np.zeros((n_episodes,2))
avg_grading = np.zeros((n_episodes,5))

# Model setting
input_size = 3
hidden_size = 30
output_size = 5
wifi_size = 520
lr = 0.001
lambda1 = lambda e: 0.9**(e/2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rl_rnn = RL_RNN(input_size, hidden_size, output_size, wifi_size, lr, device, lambda1)

# Training process
for e in range(n_episodes):
    rl_rnn.net.train()
    if e % 100 == 0:
        rl_rnn.peek_weights()
    rl_rnn.lr_sche.step()

    for i in range(len(wifi_loc_time)):
        Goal = False

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

        ## 2. Check initial KNN IoW
        while True:
            IoW_cur = IoW(wifi_loc_time[i, 520:522], coordinate, radius_gt, radius)
            if (IoW_cur == 0):
                radius *= 1.5
            elif (IoW_cur > delta):
                Goal = True
                break
            else:
                break
        if (Goal == True):
            continue

        right_action_set = []
        right_coor = coordinate.copy()
        right_radius = radius
        for t in range(5):
            right = right_action(wifi_loc_time[i, 520:522], right_coor, radius_gt, right_radius)
            right_coor, right_radius = take_action(right, right_coor, right_radius)
            right_action_set.append(right)

        input = np.concatenate((coordinate, np.array([radius])))
        action_history, coordinate, radius = rl_rnn.forward_5_rounds(wifi_loc_time[i,:520], coordinate, radius)

        avg_distance[e,0] += dis(wifi_loc_time[i, 520:522], coordinate)
        avg_iow[e,0] += IoW(wifi_loc_time[i, 520:522], coordinate, radius_gt, radius)
        avg_distance[e,1] += dis(wifi_loc_time[i, 520:522], right_coor)
        avg_iow[e,1] += IoW(wifi_loc_time[i, 520:522], right_coor, radius_gt, right_radius)
        avg_grading[e,] += rl_rnn.grading(action_history, right_action_set)

        loss = rl_rnn.back_propagation(action_history, np.array(right_action_set))
        total_loss[e] += loss

    avg_distance[e,:] /= len(wifi_loc_time)
    avg_grading[e, :] /= len(wifi_loc_time)
    avg_iow[e,:] /= len(wifi_loc_time)
    total_loss[e] /= len(wifi_loc_time)
    print("Epoch - ", e, ", avg loss :", total_loss[e], ", lr is :%.6f"%(lambda1(e)), ", avg distance:", avg_distance[e], ", avg IoW:", avg_iow[e])
    print("                                                                                         Accuracy of 5 rounds:", avg_grading[e])


np.savetxt("avg_distance.csv", avg_distance, delimiter=',')
np.savetxt("avg_iow.csv", avg_iow, delimiter=',')
np.savetxt("avg_grading.csv", avg_grading, delimiter=',')
np.savetxt("total_loss.csv", total_loss, delimiter=',')