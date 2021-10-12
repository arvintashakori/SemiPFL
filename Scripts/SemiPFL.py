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
        self.outputdim = len(self.labels_list)
        # self.data_address = r"C:\Users\walke\Documents\GitHub\SemiPFL_Wenwen\MobiNpy_4_Act"  # data adress
        self.data_address = os.path.abspath(os.path.join(
            os.getcwd(), os.pardir)) + "/Datasets/MobiNpy_4_Act/"  # data adress
        self.trial_number = 0  # which trial we use for this test
        self.label_ratio = 0.10  # ratio of labeled data
        self.eval_ratio = 0.30  # ratio of eval data
        self.number_of_client = 1  # total number of clients
        self.server_ID = [0]  # server ID
        self.batch_size = 128  # training batch size
        self.window_size = 30  # window size (for our case 30)
        self.width = 9  # data dimension (AX, AY, AZ) (GX, GY, GZ) (MX, MY, MZ)
        self.n_kernels = 16  # number of kernels for hypernetwork
        self.total_number_of_clients = 59  # total number of subjects (client + server)
        self.learning_rate = 1e-5  # learning rate for optimizer
        self.steps = 10  # total number of epochs
        self.inner_step_for_AE = 5  # number of epochs to fine tunne the Autoencoder
        self.inner_step_server_finetune = 5  # number of steps in the server side to finetune
        self.inner_step_for_model = 5  # number of steps that server fine tune its hn and user embedding parameters
        self.model_loop = False  # feedback loop for user model
        self.inner_step_for_client = 5  # number of steps that user fine tune its model
        self.inner_lr = 1e-5  # user learning rate
        self.inner_wd = 5e-5  # weight decay
        self.inout_channels = 1  # number of channels
        self.hidden = 16  # Autoencoder layer 2 parameters
        self.n_kernels_enc = 3  # autoencoder encoder kernel size
        self.hidden_dim_for_HN = 100  # hidden dimension for hypernetwork
        self.n_kernels_dec = 3  # autoencoder decoder kernel size
        self.latent_rep = 4  # latent reperesentation size
        self.n_hidden_HN = 100  # number of hidden layers in hypernetworks
        self.stride_value = 1  # stride value for autoencoder
        self.padding_value = 1  # padding value for autoencoder
        self.model_hidden_layer =128  # final model hidden layer size
        self.spec_norm = False  # True if you want to use spectral norm


def SemiPFL(params):
    # initialization
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    device = get_default_device()

    # laoding data
    nodes = Clients(address=params.data_address,
                    trial_number=params.trial_number,
                    label_ratio=params.label_ratio,
                    server_ID=params.server_ID,
                    eval_ratio=params.eval_ratio,
                    window_size=params.window_size,
                    width=params.width,
                    transform=transform,
                    num_user=params.total_number_of_clients)

    # dataloaders
    client_loader = []
    client_labeled_loaders = []
    eval_loader = []
    server_loaders = torch.utils.data.DataLoader(
        nodes.server_loaders, batch_size=params.batch_size, shuffle=True)

    for i in range(params.number_of_client):
        client_loader.append(torch.utils.data.DataLoader(
            nodes.client_loaders[i], batch_size=params.batch_size, shuffle=True))
        client_labeled_loaders.append(torch.utils.data.DataLoader(
            nodes.client_labeled_loaders[i], batch_size=params.batch_size, shuffle=True))
        eval_loader.append(torch.utils.data.DataLoader(
            nodes.eval_data[i], batch_size=params.batch_size, shuffle=True))

    # model initialization
    hnet = HN(n_nodes=params.number_of_client, embedding_dim=int(1 + params.number_of_client / 4),
              in_channels=params.inout_channels, out_dim=params.outputdim, n_kernels=params.n_kernels,
              hidden_dim=params.hidden_dim_for_HN,
              spec_norm=params.spec_norm, n_hidden=params.hidden, #params.n_hidden_HN,
              n_kernels_enc=params.n_kernels_enc,
              n_kernels_dec=params.n_kernels_dec, latent_rep=params.latent_rep, stride_value=params.stride_value,
              padding_value=params.padding_value)  # initializing the hypernetwork
    AE = Autoencoder(inout_channels=params.inout_channels, hidden=params.hidden, n_kernels_enc=params.n_kernels_enc,
                     n_kernels_dec=params.n_kernels_dec, latent_rep=params.latent_rep, stride_value=params.stride_value,
                     padding_value=params.padding_value)  # initializing the autoencoder
    model = BASEModel(latent_rep= 4 * 3 * 6, #params.width * params.window_size * params.latent_rep,
                      out_dim=params.outputdim, hidden_layer=params.model_hidden_layer)  # initilizing the base model

    # send models to device
    hnet.to(device)
    AE.to(device)
    model.to(device)

    # list of generated personalized models for each user
    client_model = []
    for i in range(params.number_of_client):
        client_model.append(model)

    # optimizer and loss functions
    optimizer = torch.optim.Adam(
        params=hnet.parameters(), lr=params.learning_rate)
    # I was using this before:  MSELoss()
    criteria_AE = torch.nn.BCEWithLogitsLoss()
    # I was using this before: CrossEntropyLoss()
    criteria_model = torch.nn.NLLLoss()

    # SemiPFL begins
    step_iter = trange(params.steps)
    results = defaultdict(list)
    for step in range(params.steps):
        hnet.train()
        # select client at random
        client_id = random.choice(range(params.number_of_client))

        # produce & load local network weights
        weights = hnet(torch.tensor([client_id], dtype=torch.long).to(device))
        AE.load_state_dict(weights)

        # init inner optimizer
        inner_optim = torch.optim.Adam(
            AE.parameters(), lr=params.inner_lr, weight_decay=params.inner_wd)

        # storing theta_i for later calculating delta theta
        inner_state = OrderedDict(
            {k: tensor.data for k, tensor in weights.items()})

        # NOTE: evaluation on sent model
        with torch.no_grad():
            AE.eval()
            prvs_loss_for_AE = 0
            for sensor_values, _ in eval_loader[client_id]:
                predicted_sensor_values = AE(sensor_values.to(device).float())
                prvs_loss_for_AE += criteria_AE(sensor_values.to(device).float(),
                                                predicted_sensor_values.to(device)).item() * sensor_values.size(0)
            prvs_loss_for_AE /= len(eval_loader[client_id].dataset)

        # inner updates -> obtaining theta_tilda
        AE.train()
        for i in range(params.inner_step_server_finetune):
            optimizer.zero_grad()
            inner_optim.zero_grad()
            for sensor_values, _ in client_loader[client_id]:
                predicted_sensor_values = AE(sensor_values.to(device).float())
                loss = criteria_AE(sensor_values.to(device).float(), predicted_sensor_values.to(device))
                loss.backward()
                inner_optim.step()
        optimizer.zero_grad()
        final_state = AE.state_dict()

        # NOTE: evaluation on sent model
        with torch.no_grad():
            AE.eval()
            prvs_loss_for_AE_updated = 0
            for sensor_values, _ in eval_loader[client_id]:
                predicted_sensor_values = AE(sensor_values.to(device).float())
                prvs_loss_for_AE_updated += criteria_AE(sensor_values.to(device).float(),
                                                        predicted_sensor_values.to(device)).item() * sensor_values.size(
                    0)
            prvs_loss_for_AE_updated /= len(eval_loader[client_id].dataset)
            AE.train()

        # calculating delta theta
        delta_theta = OrderedDict(
            {k: inner_state[k] - final_state[k] for k in weights.keys()})

        # calculating phi gradient
        hnet_grads = torch.autograd.grad(list(
            weights.values()), hnet.parameters(), grad_outputs=list(delta_theta.values()))

        # update hnet weights
        for p, g in zip(hnet.parameters(), hnet_grads):
            p.grad = g
        torch.nn.utils.clip_grad_norm_(hnet.parameters(), 50)
        optimizer.step()

        # transform the server dataset using the user autoencoder
        # train a model on  transformed dataset in server side
        if not params.model_loop:
            client_model[client_id] = model

        client_model[client_id].train()

        for i in range(params.inner_step_for_model):
            inner_optim.zero_grad()
            for sensor_values, activity in server_loaders:
                encoded_sensor_values = AE.encoder(sensor_values.to(device).float())
                predicted_activity = client_model[client_id](encoded_sensor_values)
                loss = criteria_model(predicted_activity.to(device), activity.to(device))
                loss.backward()
                inner_optim.step()

        with torch.no_grad():
            client_model[client_id].eval()
            prvs_loss_server_model = 0
            f1_score_server = 0
            for sensor_values, activity in eval_loader[client_id]:
                encoded_sensor_values = AE.encoder(sensor_values.to(device).float())
                predicted_activity = client_model[client_id](encoded_sensor_values)
                prvs_loss_server_model += criteria_model(
                    predicted_activity, activity.to(device)).item() * sensor_values.size(0)
                f1_score_server += f1_loss(activity, predicted_activity) * sensor_values.size(0)
            client_model[client_id].train()
            prvs_loss_server_model /= len(eval_loader[client_id].dataset)
            f1_score_server /= len(eval_loader[client_id].dataset)

        # fine-tune the model on user labeled dataset (I commented that since looks like its not improving)
        for param in client_model[client_id].parameters():
            param.requires_grad = False

        client_model[client_id].fc2 = nn.Linear(
            params.model_hidden_layer, params.outputdim).to(device)

        for i in range(params.inner_step_for_model):
            inner_optim.zero_grad()
            for sensor_values, activity in client_labeled_loaders[client_id]:
                encoded_sensor_values = AE.encoder(sensor_values.to(device).float())
                predicted_activity = client_model[client_id](encoded_sensor_values)
                loss = criteria_model(predicted_activity, activity.to(device))
                loss.backward()
                inner_optim.step()

        # Evaluate the model on user dataset
        with torch.no_grad():
            client_model[client_id].eval()
            prvs_loss_fine_tuned = 0
            f1_score_user = 0
            for sensor_values, activity in eval_loader[client_id]:
                encoded_sensor_values = AE.encoder(sensor_values.to(device).float())
                predicted_activity = client_model[client_id](encoded_sensor_values)
                prvs_loss_fine_tuned += criteria_model(
                    predicted_activity, activity.to(device)).item() * sensor_values.size(0)
                f1_score_user += f1_loss(activity, predicted_activity) * sensor_values.size(0)
            client_model[client_id].train()
            prvs_loss_fine_tuned /= len(eval_loader[client_id].dataset)
            f1_score_user /= len(eval_loader[client_id].dataset)
            step_iter.set_description(
                f"S:{step + 1},ID:{client_id},AE:{prvs_loss_for_AE:.4f},AE_f:{prvs_loss_for_AE_updated:.4f},Server_l:{prvs_loss_server_model:.4f},server_f1: {f1_score_server:.4f},User_l:{prvs_loss_fine_tuned:.4f},user_f1: {f1_score_user:.4f}\n")

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
