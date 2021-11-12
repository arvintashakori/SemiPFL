from SemiPFL import parameters, SemiPFL
from collections import defaultdict
import pandas as pd
import os
from utils import set_seed


if __name__ == '__main__':
    output_dim = [[11,os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "\\Datasets\\MobiNpy_V2\\"],
                  [4,os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "\\Datasets\\MobiNpy_4_Act\\"]]
    params = parameters()
    params.general_results = False  # if True calculate overal values not related to that user
    params.model_loop = False  # feedback loop for user model
    params.thr = 0.05 # feedback loop for user model
    params.server_ID = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # server ID

    f1 = defaultdict(list)
    for dim in range(1,2): # 0: 4 tasks, 1: 11 tasks
        for num_clients in range(5,51,5): # number of clients from 1 to 50
            for labeled_ratio in range(5,8): # percentage of labeled dataset in the target
                params.outputdim = output_dim[dim][0]
                params.data_address = output_dim[dim][1]
                params.number_of_client = num_clients
                params.steps = num_clients * 5
                params.label_ratio = labeled_ratio/10

                print(f'dim:{str(params.outputdim)}\n')
                print(f'num_clients:{str(params.number_of_client)}\n')
                print(f'labeled_ratio:{str(params.label_ratio)}\n')

                set_seed(params.seed)
                result = SemiPFL(params=params)
                data = pd.DataFrame.from_dict(result, orient="columns")
                pd.DataFrame.from_dict(result, orient="columns").to_csv("results//outdim_"+str(params.outputdim)+"_number_of_users_"+str(params.number_of_client)+"_labeled_ratio_"+str(params.label_ratio)+"_number_of_epochs_"+str(params.steps)+"_th_"+str(params.thr)+"_model_loop_"+str(params.model_loop)+"_general_results"+str(params.general_results)+".csv")
                user = []
                maxx = []
                total = 0
                for i in range(50):
                    user.append(data[data["Node ID"] == i]['User f1'])
                for i in range(50):
                    if len(user[i]) > 0:
                        maxx.append(max(user[i]))
                        total = total + 1

                total_new = 0
                avg = 0
                for i in range(total):
                    avg = avg + maxx[i]
                    total_new = total_new + 1

                print(f'f1:{str(avg/total_new)}\n')
                f1['out_dim'].append(params.outputdim)
                f1['number_of_users'].append(params.number_of_client)
                f1['labeled_ratio'].append(params.label_ratio)
                f1['model_loop'].append(0)
                f1['general'].append(0)
                f1['thr'].append(params.thr)
                f1['f1'].append(avg/total_new)
    pd.DataFrame.from_dict(f1, orient="columns").to_csv("results//f1.csv")

