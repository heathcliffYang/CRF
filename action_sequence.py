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
import confusion_matrix_tool as cmt
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
distances_test, indices_test = nbrs.kneighbors(wifi_loc_time_test[:,:520])

# Training setting
n_episodes = 1000
total_loss = np.zeros((n_episodes,))
radius_gt = 0.5
delta = 0.5
avg_distance = np.zeros((n_episodes,2))
avg_iow = np.zeros((n_episodes,2))
avg_grading = np.zeros((n_episodes,5))
avg_conti_score = np.zeros((n_episodes,))
batch_size = 1
action_history_batch = torch.zeros((batch_size, 5, 5))
action_label_batch = np.zeros((batch_size,5))

test_distance = np.zeros((2,))
test_iow = np.zeros((2,))
test_grading = np.zeros((5,))


# Model setting
input_size = 3
hidden_size = 32
output_size = 5
wifi_size = 520
lr = 0.005
lambda1 = lambda e: lr*(0.9**(e/1000))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rl_rnn = RL_RNN(input_size, hidden_size, output_size, wifi_size, lr, device, lambda1)
# rl_rnn.net.load_state_dict(torch.load('model.ckpt'))

for round_idx in range(1,6):
    total_loss = np.zeros((5, n_episodes,))
    avg_distance = np.zeros((5, n_episodes,2))
    avg_iow = np.zeros((5, n_episodes,2))
    avg_grading = np.zeros((5, n_episodes,5))
    avg_conti_score = np.zeros((5, n_episodes,))
    avg_round_dis = np.zeros((5, n_episodes,5))
    avg_round_rad = np.zeros((5, n_episodes,5))
    # Training process
    for e in range(n_episodes):
        rl_rnn.net.train()
        if e % 100 == 99:
            rl_rnn.peek_weights()
        rl_rnn.lr_sche.step()

        y_true = np.zeros((len(wifi_loc_time),5))
        y_pred = np.zeros((len(wifi_loc_time),5))
        for i in range(len(wifi_loc_time)):
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
                else:
                    break

            orig_coordinate = coordinate.copy()
            orig_radius = radius
            right_action_set = []
            right_coor = coordinate.copy()
            right_radius = radius
            for t in range(5):
                right = right_action(wifi_loc_time[i, 520:522], right_coor, radius_gt, right_radius)
                right_coor, right_radius = take_action(right, right_coor, right_radius)
                right_action_set.append(right)

            input = np.concatenate((coordinate, np.array([radius])))
            # ver. 1
            action_history, coordinate, radius = rl_rnn.forward_5_rounds(wifi_loc_time[i,:520], coordinate, radius, round_idx)
            # ver. 2
            # action_history, coordinate, radius = rl_rnn.forward_down(wifi_loc_time[i,:520], coordinate, radius)

            avg_distance[round_idx-1,e,0] += dis(wifi_loc_time[i, 520:522], coordinate)
            avg_iow[round_idx-1,e,0] += IoW(wifi_loc_time[i, 520:522], coordinate, radius_gt, radius)
            avg_distance[round_idx-1,e,1] += dis(wifi_loc_time[i, 520:522], right_coor)
            avg_iow[round_idx-1,e,1] += IoW(wifi_loc_time[i, 520:522], right_coor, radius_gt, right_radius)
            grading_sheet, ans_sheet = rl_rnn.grading(action_history, right_action_set)
            avg_grading[round_idx-1,e,] += grading_sheet
            avg_conti_score[round_idx-1,e] += rl_rnn.conti_success_score(grading_sheet)


            if batch_size != 1:
                print(right_action_set)
                batch_iter = i % batch_size
                if batch_iter != batch_size -1:
                    action_history_batch[batch_iter] = action_history
                    action_label_batch[batch_iter] = np.array(right_action_set)
                else:
                    loss = rl_rnn.back_propagation(action_history_batch, action_label_batch, round_idx)
                    total_loss[round_idx-1,e] += loss
            else:
                loss = rl_rnn.back_propagation(action_history, np.array(right_action_set), round_idx)
                total_loss[round_idx-1,e] += loss

            # confusion_matrix_stack
            y_true[i] = np.array(right_action_set)
            y_pred[i] = ans_sheet

            # compute distance after each of 5 rounds & radius
            for k in range(5):
                orig_coordinate, orig_radius = take_action(ans_sheet[k], orig_coordinate, orig_radius)
                avg_round_dis[round_idx-1, e, k] += dis(wifi_loc_time[i, 520:522], orig_coordinate)
                avg_round_rad[round_idx-1, e, k] += orig_radius

        

        avg_round_dis[round_idx-1,e,:] /= len(wifi_loc_time)
        avg_round_rad[round_idx-1,e,:] /= len(wifi_loc_time)
        avg_distance[round_idx-1,e,:] /= len(wifi_loc_time)
        avg_grading[round_idx-1,e, :] #/= len(wifi_loc_time)
        avg_iow[round_idx-1,e,:] /= len(wifi_loc_time)
        avg_conti_score[round_idx-1,e] /= len(wifi_loc_time)
        print("Epoch -", e, round_idx,"avg loss :", total_loss[round_idx-1,e], ", lr is :", lambda1(e), ", avg distance:", avg_distance[round_idx-1,e], "\navg IoW:", avg_iow[round_idx-1,e], ", avg cont_score:", avg_conti_score[round_idx-1,e])
        print("Accuracy of 5 rounds:", avg_grading[round_idx-1,e])
        print("Round dis: ", avg_round_dis[round_idx-1,e])
        print("Round rad: ", avg_round_rad[round_idx-1,e])

        if round_idx == 5:
            for l in range(5):
                cmt.plot_confusion_matrix(y_true[:,l], y_pred[:,l], classes=['R'+str(i) for i in range(5)], normalize=False, title='Normalized confusion matrix')
                plt.savefig("cm/Conf_matrix_e_{}_round_{}_same_w.png".format(e, l))
                plt.close()
                plt.cla()
                plt.clf()

        #Testing
        rl_rnn.net.eval()
        test_distance = np.zeros((2,))
        test_iow = np.zeros((2,))
        test_grading = np.zeros((5,))
        test_conti_score = 0
        for i in range(len(wifi_loc_time_test)):
            Hx = 0.
            Hy = 0.
            for m in range(3):
                Hx += wifi_loc_time[indices_test[i, m], 520]
                Hy += wifi_loc_time[indices_test[i, m], 521]
            Hx /= 3.
            Hy /= 3.
            coordinate = np.array([Hx, Hy])
            radius = 1.

            while True:
                IoW_cur = IoW(wifi_loc_time_test[i, 520:522], coordinate, radius_gt, radius)
                if (IoW_cur == 0):
                    radius *= 1.5
                else:
                    break

            right_action_set = []
            right_coor = coordinate.copy()
            right_radius = radius
            for t in range(5):
                right = right_action(wifi_loc_time_test[i, 520:522], right_coor, radius_gt, right_radius)
                right_coor, right_radius = take_action(right, right_coor, right_radius)
                right_action_set.append(right)

            input = np.concatenate((coordinate, np.array([radius])))
            action_history, coordinate, radius = rl_rnn.forward_down(wifi_loc_time_test[i,:520], coordinate, radius)

            test_distance[0] += dis(wifi_loc_time_test[i, 520:522], coordinate)
            test_iow[0] += IoW(wifi_loc_time_test[i, 520:522], coordinate, radius_gt, radius)
            test_distance[1] += dis(wifi_loc_time_test[i, 520:522], right_coor)
            test_iow[1] += IoW(wifi_loc_time_test[i, 520:522], right_coor, radius_gt, right_radius)
            grading_sheet, _ = rl_rnn.grading(action_history, right_action_set)
            test_grading += grading_sheet
            test_conti_score += rl_rnn.conti_success_score(grading_sheet)

        print("Test distance: {}, iow: {}\ngrading:{}, conti_score:{}\n".format(test_distance/len(wifi_loc_time_test), test_iow/len(wifi_loc_time_test), test_grading/len(wifi_loc_time_test), test_conti_score/len(wifi_loc_time_test)))
        
    torch.save(rl_rnn.net.state_dict(), 'model_round_'+str(round_idx)+'_same_w.ckpt')

# Plot
for i in range(5):
    plt.plot([w for w in range(n_episodes)], avg_distance[i,:,0], 'r-', label='model')
    plt.plot([w for w in range(n_episodes)], avg_distance[i,:,1], 'b-', label='right')
plt.title("Distance_round ")
plt.legend()
plt.savefig("Distance.png")
plt.clf()

for i in range(5):
    plt.plot([w for w in range(n_episodes)], avg_iow[i,:,0], 'r-', label='model')
    plt.plot([w for w in range(n_episodes)], avg_iow[i,:,1], 'b-', label='right')
plt.title("IoW_round ")
plt.legend()
plt.savefig("IoW.png")
plt.clf()

for i in range(5):
    for k in range(5):
        plt.plot([w for w in range(n_episodes)], avg_grading[i,:,k], label='Round '+str(k+1))
plt.title("Hit rate of each round_round ")
plt.legend()
plt.savefig("Grading.png")
plt.clf()

for i in range(5):
    plt.plot([w for w in range(n_episodes)], avg_conti_score[i])
plt.title("Continuous hit score_round ")
plt.savefig("conti.png")
plt.clf()

for i in range(5):
    for k in range(5):
        plt.plot([w for w in range(n_episodes)], avg_round_dis[i,:,k], label='Round '+str(k+1))
plt.title("Distance after each round_round ")
plt.legend()
plt.savefig("Round_dis.png")
plt.clf()

for i in range(5):
    for k in range(5):
        plt.plot([w for w in range(n_episodes)], avg_round_rad[i,:,k], label='Round '+str(k+1))
plt.title("Radius after each round_round ")
plt.legend()
plt.savefig("Round_rad.png")
plt.clf()