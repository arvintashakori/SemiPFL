import pandas as pd
from collections import defaultdict
import os

f1 = defaultdict(list)
output_dim = [[11, os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "\\Datasets\\MobiNpy_V2\\"],
              [4, os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "\\Datasets\\MobiNpy_4_Act\\"]]

for dim in range(2): # 1: 4 tasks, 0: 11 tasks
    for num_clients in range(5,55,5): # number of clients from 1 to 50
        for labeled_ratio in range(1,8): # percentage of labeled dataset in the target
            file_name = "results//outdim_"+str(output_dim[dim][0])+"_number_of_users_"+str(num_clients)+"_labeled_ratio_"+str(labeled_ratio/10)+"_number_of_epochs_"+str(num_clients*5)+"_th_"+str(0.05)+"_model_loop_False_general_resultsFalse.csv"
            data = pd.read_csv(file_name)
            user = []
            maxx = []
            total = 0
            for i in range(50):
                user.append(data[data["Node ID"] == i]['User f1'])
            for i in range(50):
                if len(user[i]) > 2:
                    maxx.append(max(user[i]))
                    total = total + 1;

            total_new = 0
            avg = 0
            std = 0
            for i in range(total):
                if dim == 0:
                    if maxx[i]>0.75:
                        avg = avg + maxx[i]
                        total_new = total_new + 1
                if dim == 1:
                    if maxx[i] > 0.90:
                        avg = avg + maxx[i]
                        total_new = total_new + 1

            for i in range(total):
                if dim == 0:
                    if maxx[i]>0.75:
                        std = std = pow(maxx[i]-avg,2)

                if dim == 1:
                    if maxx[i] > 0.90:
                        std = std = pow(maxx[i] - avg,2)

            std = pow(std,0.5)

            #print(f'f1:{str(avg / total_new)}\n')
            f1['out_dim'].append(dim)
            f1['number_of_users'].append(num_clients)
            f1['labeled_ratio'].append(labeled_ratio/10)
            f1['model_loop'].append(0)
            f1['general'].append(0)
            f1['thr'].append(0.05)
            f1['f1'].append(avg / total_new)
            f1['std'].append(std / (total_new))
            f1['activeusers'].append(total_new)
            for i in range(50):
                if dim == 0:
                    if i < total:
                        if maxx[i]>0.75:
                            f1['user'+str(i)].append(maxx[i])
                        else:
                            f1['user' + str(i)].append(0)
                    else:
                        f1['user' + str(i)].append(0)
                if dim == 1:
                    if i < total:
                        if maxx[i] > 0.90:
                            f1['user'+str(i)].append(maxx[i])
                        else:
                            f1['user' + str(i)].append(0)
                    else:
                        f1['user' + str(i)].append(0)

        pd.DataFrame.from_dict(f1, orient="columns").to_csv("results//f1_bynow.csv")