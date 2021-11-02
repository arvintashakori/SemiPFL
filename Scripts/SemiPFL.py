from models import HN, Autoencoder, BASEModel
from collections import OrderedDict, defaultdict
from utils import get_default_device, set_seed, f1_loss
from node import Clients

import os
from tqdm import trange
import pandas as pd
import random

import torch
from torch import nn
import torch.utils.data
from torchvision import transforms


class parameters:
    def __init__(self):
        self.seed = 0
        self.labels_list = ['JOG', 'JUM', 'STD', 'WAL']  # list of activities
        # self.outputdim = len(self.labels_list)
        self.outputdim = 11
        # self.data_address = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "\\Datasets\\MobiNpy_4_Act\\"  # data adress
        self.data_address = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "\\Datasets\\MobiNpy_V2\\"  # data adress
        self.trial_number = 0  # which trial we use for this test
        self.label_ratio = 0.1  # ratio of labeled data
        self.eval_ratio = 0.30  # ratio of eval data
        self.number_of_client = 50  # total number of clients
        self.server_ID = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # server ID
        self.batch_size = 128  # training batch size
        self.window_size = 30  # window size (for our case 30)
        self.width = 9  # data dimension (AX, AY, AZ) (GX, GY, GZ) (MX, MY, MZ)
        self.total_number_of_clients = 59  # total number of subjects (client + server)
        self.learning_rate = 1e-3  # learning rate for optimizer
        self.steps = 200  # total number of epochs
        self.inner_step_for_AE = 5  # number of epochs to fine tunne the Autoencoder
        self.inner_step_for_model = 5  # number of steps that server fine tune its model for user
        self.model_loop = False # feedback loop for user model
        self.inner_step_for_client = 5  # number of steps that user fine tune its model
        self.inner_lr = 1e-3  # user learning rate
        self.inner_wd = 5e-5  # weight decay
        self.hidden_dim_for_HN = 10  # hidden dimension for hypernetwork
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        self.device = get_default_device()
        # Autoencoder and base model parameters
        self.AE_layer_1 = 256 # Autoencoder hidden layer 1 size
        self.AE_layer_2 = 128 # Autoencoder hidden layer 2 size
        self.latent_rep = 64 # latent reperesentation size
        self.base_model_hidden_layer = 16 # base model hidden layer size
        self.thr = 0.01 # threshould to find similar datapoints for user
        self.general_results = False # if True calculate overal values not related to that user

def AE_evaluate(client_AE, eval_loader, number_of_clients, criteria_AE, general_results, params, id = 0):
    with torch.no_grad():
        total_loss = 0
        if general_results:
            for client_id in range(number_of_clients):
                prvs_loss_for_AE = 0
                for sensor_values, _ in eval_loader[client_id]:
                    client_AE[client_id].eval()
                    sensor_values = sensor_values.view(sensor_values.size(0), -1)
                    predicted_sensor_values = client_AE[client_id](sensor_values.to(params.device).float())
                    prvs_loss_for_AE += criteria_AE(predicted_sensor_values.to(params.device),
                                                    sensor_values.to(params.device).float()).item() * sensor_values.size(0)
                prvs_loss_for_AE /= len(eval_loader[client_id].dataset)
                total_loss += prvs_loss_for_AE/number_of_clients
        else:
            client_AE[id].eval()
            prvs_loss_for_AE = 0
            for sensor_values, _ in eval_loader[id]:
                sensor_values = sensor_values.view(sensor_values.size(0), -1)
                predicted_sensor_values = client_AE[id](sensor_values.to(params.device).float())
                prvs_loss_for_AE += criteria_AE(predicted_sensor_values.to(params.device),
                                                sensor_values.to(params.device).float()).item() * sensor_values.size(0)
            total_loss = prvs_loss_for_AE / len(eval_loader[id].dataset)

        return total_loss


def model_evaluate(client_AE, client_model, eval_loader, number_of_clients, criteria_model, general_results, params, id = 0):
    with torch.no_grad():
        total_loss = 0
        total_f1 = 0
        if general_results:
            for client_id in range(number_of_clients):
                client_model[client_id].eval()
                client_AE[client_id].eval()
                prvs_loss_server_model = 0
                f1_score_server = 0
                for sensor_values, activity in eval_loader[client_id]:
                    sensor_values = sensor_values.view(sensor_values.size(0), -1)
                    encoded_sensor_values = client_AE[client_id].encoder(sensor_values.to(params.device).float())
                    predicted_activity = client_model[client_id](encoded_sensor_values)
                    prvs_loss_server_model += criteria_model(predicted_activity,
                                                             activity.to(params.device)).item() * sensor_values.size(0)
                    f1_score_server += f1_loss(activity, predicted_activity) * sensor_values.size(0)
                prvs_loss_server_model /= len(eval_loader[client_id].dataset)
                f1_score_server /= len(eval_loader[client_id].dataset)
                total_loss += prvs_loss_server_model / number_of_clients
                total_f1 += f1_score_server / number_of_clients
        else:
            client_model[id].eval()
            prvs_loss_server_model = 0
            f1_score_server = 0
            for sensor_values, activity in eval_loader[id]:
                sensor_values = sensor_values.view(sensor_values.size(0), -1)
                encoded_sensor_values = client_AE[id].encoder(sensor_values.to(params.device).float())
                predicted_activity = client_model[id](encoded_sensor_values)
                prvs_loss_server_model += criteria_model(predicted_activity,
                                                         activity.to(params.device)).item() * sensor_values.size(0)
                f1_score_server += f1_loss(activity, predicted_activity) * sensor_values.size(0)
            total_loss = prvs_loss_server_model / len(eval_loader[id].dataset)
            total_f1 = f1_score_server / len(eval_loader[id].dataset)

        return total_loss, total_f1


def SemiPFL(params):

    # model initialization
    hnet = HN(n_nodes = params.number_of_client,
              embedding_dim = int(1 + params.number_of_client / 4),
              hidden_dim = 100,
              n_hidden = 10)  # initializing the hypernetwork

    AE = Autoencoder(inout_dim = params.width * params.window_size,
                     layer1 = params.AE_layer_1,
                     layer2 = params.AE_layer_2,
                     latent_rep = params.latent_rep)  # initializing the autoencoder

    model = BASEModel(latent_rep=params.latent_rep,
                      out_dim=params.outputdim,
                      hidden_layer=params.base_model_hidden_layer)  # initilizing the base model

    # send models to device
    hnet.to(params.device)
    AE.to(params.device)
    model.to(params.device)


    # loading data
    nodes = Clients(address=params.data_address,
                    trial_number=params.trial_number,
                    label_ratio=params.label_ratio,
                    server_ID=params.server_ID,
                    eval_ratio=params.eval_ratio,
                    window_size=params.window_size,
                    width=params.width,
                    transform=params.transform,
                    num_user=params.total_number_of_clients)

    # initializing dataloaders
    client_loader = []
    client_labeled_loaders = []
    eval_loader = []
    server_loaders = torch.utils.data.DataLoader(nodes.server_loaders,
                                                 batch_size=params.batch_size,
                                                 shuffle=True)

    for i in range(params.number_of_client):
        client_loader.append(torch.utils.data.DataLoader(nodes.client_loaders[i],
                                                         batch_size=params.batch_size,
                                                         shuffle=True))
        client_labeled_loaders.append(torch.utils.data.DataLoader(nodes.client_labeled_loaders[i],
                                                                  batch_size=params.batch_size,
                                                                  shuffle=True))
        eval_loader.append(torch.utils.data.DataLoader(nodes.eval_data[i],
                                                       batch_size=params.batch_size,
                                                       shuffle=True))


    # list of generated personalized models for each user
    client_model = []
    client_AE = []
    for i in range(params.number_of_client):
        client_model.append(model)
        client_AE.append(AE)

    # optimizer and loss functions
    optimizer_hnet = torch.optim.Adam(params=hnet.parameters(),
                                      lr=params.learning_rate)

    criteria_AE = torch.nn.MSELoss()
    criteria_model = torch.nn.NLLLoss()

    # SemiPFL begins
    step_iter = trange(params.steps)
    results = defaultdict(list)
    for step in range(params.steps):
        hnet.train()
        # select client at random
        client_id = random.choice(range(params.number_of_client))

        # produce & load local network weights
        weights = hnet(torch.tensor([client_id], dtype=torch.long).to(params.device))
        client_AE[client_id].load_state_dict(weights)

        # init inner optimizer
        optimizer_AE = torch.optim.Adam(AE.parameters(), lr=params.inner_lr)

        # storing theta_i for later calculating delta theta
        inner_state = OrderedDict({k: tensor.data for k, tensor in weights.items()})

        # NOTE: evaluation on sent model
        prvs_loss_for_AE = AE_evaluate(client_AE = client_AE,
                                       eval_loader = eval_loader,
                                       number_of_clients = params.number_of_client,
                                       criteria_AE = criteria_AE,
                                       general_results = params.general_results,
                                       id= client_id,
                                       params = params)

        # Step 2: inner updates -> obtaining theta_tilda
        client_AE[client_id].train()
        for i in range(params.inner_step_for_AE):
            optimizer_hnet.zero_grad()
            optimizer_AE.zero_grad()
            for sensor_values, _ in client_loader[client_id]:
                sensor_values = sensor_values.view(sensor_values.size(0), -1)
                predicted_sensor_values = client_AE[client_id](sensor_values.to(params.device).float())
                loss = criteria_AE(predicted_sensor_values.to(params.device).float(),sensor_values.to(params.device).float())
                loss.backward()
                optimizer_AE.step()
            #print('AE model epoch [{}/{}], loss:{:.4f}'.format(i + 1, params.inner_step_server_finetune, loss.data))



        # NOTE: evaluation on sent model
        prvs_loss_for_AE_updated = AE_evaluate(client_AE=client_AE,
                                       eval_loader=eval_loader,
                                       number_of_clients=params.number_of_client,
                                       criteria_AE=criteria_AE,
                                       general_results=params.general_results,
                                       id=client_id,
                                       params = params)

        # update hnet
        optimizer_hnet.zero_grad()
        final_state = client_AE[client_id].state_dict()

        # calculating delta theta
        delta_theta = OrderedDict({k: inner_state[k] - final_state[k] for k in weights.keys()})

        # calculating ds = tphi gradient
        hnet_grads = torch.autograd.grad(list(weights.values()), hnet.parameters(), grad_outputs=list(delta_theta.values()))

        # update hnet weights
        for p, g in zip(hnet.parameters(), hnet_grads):
            p.grad = g
        torch.nn.utils.clip_grad_norm_(hnet.parameters(), 50)
        optimizer_hnet.step()



        # transform the server dataset using the user autoencoder
        # train a model on  transformed dataset in server side
        if not params.model_loop:
            client_model[client_id] = model

        client_model[client_id].train()
        client_AE[client_id].eval()
        # init base optimizer
        optimizer_base = torch.optim.Adam(client_model[client_id].parameters(), lr=params.inner_lr)

        for i in range(params.inner_step_for_model):
            optimizer_base.zero_grad()
            for sensor_values, activity in server_loaders:
                sensor_values = sensor_values.view(sensor_values.size(0), -1)
                # check if the sample is similar to user data
                if criteria_AE(AE(sensor_values.to(params.device).float()),sensor_values.to(params.device).float()).item() < params.thr:
                    encoded_sensor_values = client_AE[client_id].encoder(sensor_values.to(params.device).float())
                    predicted_activity = client_model[client_id](encoded_sensor_values)
                    loss = criteria_model(predicted_activity.to(params.device), activity.to(params.device))
                    loss.backward()
                    optimizer_base.step()
            #print('model epoch [{}/{}], loss:{:.4f}'.format(i + 1, params.inner_step_for_model, loss.data))

        # Evaluate the model on user dataset
        prvs_loss_server_model, f1_score_server = model_evaluate(client_AE = client_AE,
                                                                 client_model = client_model,
                                                                 eval_loader = eval_loader,
                                                                 number_of_clients = params.number_of_client,
                                                                 criteria_model = criteria_model,
                                                                 general_results = params.general_results,
                                                                 id = client_id,
                                                                 params = params)


        client_model[client_id].train()
        client_AE[client_id].eval()
        #for param in client_model[client_id].parameters():
         #   param.requires_grad = False

        #client_model[client_id].fc2 = nn.Linear(params.base_model_hidden_layer, params.outputdim).to(params.device)

        for i in range(params.inner_step_for_client):
            optimizer_base.zero_grad()
            for sensor_values, activity in client_labeled_loaders[client_id]:
                sensor_values = sensor_values.view(sensor_values.size(0), -1)
                encoded_sensor_values = client_AE[client_id].encoder(sensor_values.to(params.device).float())
                predicted_activity = client_model[client_id](encoded_sensor_values)
                loss = criteria_model(predicted_activity, activity.to(params.device))
                loss.backward()
                optimizer_base.step()
            #print('modelfine epoch [{}/{}], loss:{:.4f}'.format(i + 1, params.inner_step_for_client, loss.data))

        # Evaluate the model on user dataset
        prvs_loss_fine_tuned, f1_score_user = model_evaluate(client_AE=client_AE,
                                                                 client_model=client_model,
                                                                 eval_loader=eval_loader,
                                                                 number_of_clients=params.number_of_client,
                                                                 criteria_model=criteria_model,
                                                                 general_results=params.general_results,
                                                                 id=client_id,
                                                                 params = params)


        step_iter.set_description(f"S:{step + 1},ID:{client_id},AE:{prvs_loss_for_AE:.4f},AE_f:{prvs_loss_for_AE_updated:.4f},Server_l:{prvs_loss_server_model:.4f},server_f1: {100*f1_score_server:.2f},User_l:{prvs_loss_fine_tuned:.4f},user_f1: {100*f1_score_user:.4f}\n")

        # save results
        results['Step'].append(step + 1)
        results['Node ID'].append(client_id)
        results['AE loss'].append(prvs_loss_for_AE)
        results['Server model loss'].append(prvs_loss_server_model)
        results['Client fine tuned loss'].append(prvs_loss_fine_tuned)
        results['AE fine tuned loss'].append(prvs_loss_for_AE_updated)
        results['Server f1'].append(f1_score_server.item())
        results['User f1'].append(f1_score_user.item())

    return results


if __name__ == '__main__':
    params = parameters()
    set_seed(params.seed)
    result = SemiPFL(params=params)
    pd.DataFrame.from_dict(result, orient="columns").to_csv("results.csv")
