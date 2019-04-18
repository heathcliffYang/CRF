import numpy as np 
import math
import csv

def dataloader(floor, building, filepath="1478167720_9233432_trainingData.csv"):
    ## Count the number of data points in building id & floor id
        data_num = 0
        with open(filepath, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                if (row[523] == 'BUILDINGID'):
                    continue
                elif (int(row[523]) is not building or int(row[522]) is not floor):
                    continue
                data_num += 1
        print(data_num)
        ## if there are no data, continue to next floor 
        if (data_num == 0):
            raise EOFError
            
        ## Load data points in
        wifi_loc_time = np.zeros(shape = (data_num, 524))
        i=-1
        with open("1478167720_9233432_trainingData.csv", newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                if (row[523] == 'BUILDINGID'):
                    continue
                elif (int(row[523]) is not building or int(row[522]) is not floor):
                    continue
                i = i+1
                if (i > data_num):
                    break
                # wifi
                wifi_loc_time[i-1][:520] = np.array(row[:520])
                # location x, y
                wifi_loc_time[i-1][520:522] = np.array(row[520:522])
                # userID
                wifi_loc_time[i-1][522] = np.array(row[526])
                # time stamp
                wifi_loc_time[i-1][-1] = np.array(row[-1])
        
        ## Sort by time stamp
        return wifi_loc_time[wifi_loc_time[:,-1].argsort()]


def IoW(gt, coor, radius_gt, radius):
    ### IoW cross section calculation
#     print("s-1")
    cross_list_0 = [(gt[0]-radius_gt, 0), (gt[0]+radius_gt, 0)]
    cross_list_1 = [(gt[1]-radius_gt, 0), (gt[1]+radius_gt, 0)]
#     print("s-2")
    cross_list_0.append((coor[0]-radius, 1))
    cross_list_0.append((coor[0]+radius, 1))
    cross_list_1.append((coor[1]-radius, 1))
    cross_list_1.append((coor[1]+radius, 1))
#     print("s-3")                 
    cross_list_0.sort(key = lambda x : x[0])
    cross_list_1.sort(key = lambda x : x[0])
#     print("s-4") 
    if (cross_list_0[0][1] != cross_list_0[1][1] and cross_list_1[0][1] != cross_list_1[1][1]):
        return (cross_list_0[2][0] - cross_list_0[1][0]) * (cross_list_1[2][0] - cross_list_1[1][0]) / ((radius*2)**2)
    else:
        return 0.
    
    
def distance_progress(coor_gt, coor_cur, coor_next):
    dis_cur = math.sqrt((coor_gt[0]-coor_cur[0])**2 + (coor_gt[1]-coor_cur[1])**2)
    dis_next = math.sqrt((coor_gt[0]-coor_next[0])**2 + (coor_gt[1]-coor_next[1])**2)
    return dis_cur - dis_next

def dis(coor_gt, coor_cur):
    dis_cur = math.sqrt((coor_gt[0]-coor_cur[0])**2 + (coor_gt[1]-coor_cur[1])**2)
    return dis_cur


def right_action(coor_gt, coor_cur, radius_gt, radius):
    next_coordinate = coor_cur.copy()
    action = 0
    next_coordinate[0] -= radius/2.
    next_coordinate[1] += radius/2.
    area = IoW(coor_gt, next_coordinate, radius_gt, radius*0.7)

    next_coordinate = coor_cur.copy()
    next_coordinate[0] += radius/2.
    next_coordinate[1] -= radius/2.
    a = IoW(coor_gt, next_coordinate, radius_gt, radius*0.7)
    if area < a:
        action = 1
        area = a

    next_coordinate = coor_cur.copy()
    next_coordinate[0] -= radius/2.
    next_coordinate[1] -= radius/2.
    a = IoW(coor_gt, next_coordinate, radius_gt, radius*0.7)
    if area < a:
        action = 2
        area = a
    
    next_coordinate = coor_cur.copy()
    next_coordinate[0] += radius/2.
    next_coordinate[1] += radius/2.
    a = IoW(coor_gt, next_coordinate, radius_gt, radius*0.7)
    if area < a:
        action = 3
        area = a

    next_coordinate = coor_cur.copy()
    a = IoW(coor_gt, next_coordinate, radius_gt, radius*.7)
    if area < a:
        action = 4
        area = a
    return action