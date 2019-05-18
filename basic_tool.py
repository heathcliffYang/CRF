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

def take_action(action, coordinate, radius):
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
    radius *= 0.7
    return next_coordinate, radius

def right_action(coor_gt, coor_cur, radius_gt, radius):
    next_coordinate, next_radius = take_action(0, coor_cur, radius)
    max_area = IoW(coor_gt, next_coordinate, radius_gt, next_radius)
    best_action = 0
    for i in range(1,5):
        next_coordinate, next_radius = take_action(i, coor_cur, radius)
        a = IoW(coor_gt, next_coordinate, radius_gt, next_radius)
        if max_area < a:
            best_action = i
            max_area = a

    return best_action

def balance_dataset(fn="right_actions/right_action_log_2_0.csv"):
    with open(fn, newline='') as csvfile:
        right_action_log = []
        rows = csv.reader(csvfile)
        for row in rows:
            line = []
            for num in row:
                line.append(float(num))
            right_action_log.append(line)

        right_action_log = np.array(right_action_log, dtype=int).T

    pairs =np.zeros((5, 2, 5),dtype=int) ## unique and count
    minor_list = np.array([x for x in range(right_action_log.shape[1])])
    for i in range(5):
        pairs[i,0], pairs[i,1] = np.unique(right_action_log[i], return_counts=True)
        pairs[i,0] = pairs[i,0, pairs[i,1,:].argsort()]
        pairs[i,1] = pairs[i,1, pairs[i,1,:].argsort()]
        minor = np.where(right_action_log[i] != pairs[i,0,-1])
        minor_list = np.concatenate((minor_list, minor[0]))
        print("round",i,pairs[i,0],pairs[i,1], minor[0].shape)

    print(len(minor_list))

    return minor_list


    # pick up minor class

    
    


