import os
from models import HN, Autoencoder, BASEModel
from collections import OrderedDict, defaultdict
from node import Clients
from torch import nn
from tqdm import trange
import torch
import pandas as pd
import random
import numpy as np
import torch.utils.data
from torchvision import transforms
from sklearn.metrics import f1_score


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def f1_loss(y_true: torch.Tensor, y_pred: torch.Tensor, is_training=False) -> torch.Tensor:
    _, pred = torch.max(y_pred, dim=1)
    f1 = f1_score(y_true.cpu(), pred.cpu(), average='weighted')
    return f1



class parameters:
    def __init__(self):
        self.seed = 0
        self.labels_list = ['JOG', 'JUM',  'STD', 'WAL']  # list of activities
        self.outputdim = 4 # should be len(self.labels_list)
        self.data_address = os.path.abspath(os.path.join(
            os.getcwd(), os.pardir)) + "/Datasets/"  # data adress
        self.trial_number = 0  # which trial we use for this test
        self.label_ratio = 0.20  # ratio fo labeled data
        self.number_of_client = 58  # total number of clients
        self.server_ID = 0  # server ID
        self.batch_size = 128  # training batch size
        self.window_size = 30  # window size (for our case 30)
        self.width = 9  # data dimension (AX, AY, AZ) (GX, GY, GZ) (MX, MY, MZ)
        self.n_kernels = 16  # number of kernels for hypernetwork
        self.device = 'cuda' # device which we run the simulation use 'cuda' if gpu available otherwise 'cpu'
        self.total_number_of_clients = 59 # total number of subjects (client + server)
        self.learning_rate = 1e-3  # learning rate for optimizer
        self.steps = 1000  # total number of epochs
        self.inner_step_for_AE = 1000  # number of steps to fine tunne the Autoencoder
        # number of steps in the server side to finetune
        self.inner_step_server_finetune = 50
        # number of steps that server fine tune its hn and user embedding parameters
        self.inner_step_for_server = 1000
        self.inner_step_for_client = 1000  # number of steps that user fine tune its model
        self.inner_lr = 1e-3  # user learning rate
        self.inner_wd = 5e-5  # weight decay
        self.inout_channels = 1  # number of channels
        self.hidden = 16  # Autoencoder layer 2 parameters
        self.n_kernels_enc = 3  # autoencoder encoder kernel size
        self.hidden_dim_for_HN = 100  # hidden dimension for hypernetwork
        self.n_kernels_dec = 3  # autoencoder decoder kernel size
        self.latent_rep = 16  # latent reperesentation size
        self.n_hidden_HN = 1  # number of hidden layers in hypernetworks
        self.stride_value = 1  # stride value for autoencoder
        self.padding_value = 1  # padding value for autoencoder
        self.model_hidden_layer = 128  # final model hidden layer size
        self.spec_norm = False  # True if you want to use spectral norm


def SemiPFL(params):
    # initialization
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    device = torch.device(params.device)

    # laoding data
    nodes = Clients(address=params.data_address,
                    trial_number=params.trial_number,
                    label_ratio=params.label_ratio,
                    server_ID=params.server_ID,
                    window_size=params.window_size,
                    width=params.width,
                    transform=transform,
                    num_user=params.total_number_of_clients)

    # dataloaders
    client_loader = []
    client_labeled_loaders = []
    server_loaders = torch.utils.data.DataLoader(nodes.server_loaders, batch_size=params.batch_size, shuffle=True)

    for i in range(params.number_of_client):
        client_loader.append(torch.utils.data.DataLoader(nodes.client_loaders[i], batch_size=params.batch_size, shuffle=True))
        client_labeled_loaders.append(torch.utils.data.DataLoader(nodes.client_labeled_loaders[i], batch_size=params.batch_size, shuffle=True))

    # model initialization

    hnet = HN(n_nodes=params.number_of_client, embedding_dim=int(1 + params.number_of_client / 4),
              in_channels=params.inout_channels, out_dim=params.outputdim, n_kernels=params.n_kernels, hidden_dim=params.hidden_dim_for_HN,
              spec_norm=params.spec_norm, n_hidden=params.n_hidden_HN, n_kernels_enc=params.n_kernels_enc, n_kernels_dec=params.n_kernels_dec, latent_rep=params.latent_rep, stride_value=params.stride_value, padding_value=params.padding_value)  # initializing the hypernetwork
    AE = Autoencoder(inout_channels=params.inout_channels, hidden=params.hidden, n_kernels_enc=params.n_kernels_enc,
                     n_kernels_dec=params.n_kernels_dec, latent_rep=params.latent_rep, stride_value=params.stride_value, padding_value=params.padding_value)  # initializing the autoencoder
    model = BASEModel(latent_rep=params.width * params.window_size * params.latent_rep,
                      out_dim=params.outputdim, hidden_layer=params.model_hidden_layer)  # initilizing the base model
    # ****** TO WENWEN: out_dim is not 20 it should be len(params.labels_list)

    # send models to device
    hnet.to(device)
    AE.to(device)
    model.to(device)

    # list of generated personalized models for each user
    client_model = []
    for i in range(params.total_number_of_clients):
        client_model.append(model)

    # optimizer and loss functions
    optimizer = torch.optim.Adam(
        params=hnet.parameters(), lr=params.learning_rate)
    # I was using this before:  MSELoss()
    criteria_AE = torch.nn.BCEWithLogitsLoss()
    # I was using this before: CrossEntropyLoss()
    criteria_model = torch.nn.NLLLoss()

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
        inner_optim = torch.optim.Adam(AE.parameters(), lr=params.inner_lr, weight_decay=params.inner_wd)

        # storing theta_i for later calculating delta theta
        inner_state = OrderedDict({k: tensor.data for k, tensor in weights.items()})

        # NOTE: evaluation on sent model
        with torch.no_grad():
            AE.eval()
            batch = next(iter(client_loader[client_id]))
            sensor_values, _ = tuple(t.to(device) for t in batch)
            predicted_sensor_values = AE(sensor_values.float())
            prvs_loss_for_AE = criteria_AE(sensor_values.float(), predicted_sensor_values)
            AE.train()

        # inner updates -> obtaining theta_tilda
        for i in range(params.inner_step_server_finetune):
            AE.train()
            inner_optim.zero_grad()
            optimizer.zero_grad()
            batch = next(iter(client_loader[client_id]))
            sensor_values, _ = tuple(t.to(device) for t in batch)
            predicted_sensor_values = AE(sensor_values.float())
            loss = criteria_AE(sensor_values.float(), predicted_sensor_values)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(AE.parameters(), 50)
            inner_optim.step()
        optimizer.zero_grad()
        final_state = AE.state_dict()

        # NOTE: evaluation on sent model
        with torch.no_grad():
            AE.eval()
            batch = next(iter(client_loader[client_id]))
            sensor_values, _ = tuple(t.to(device) for t in batch)
            predicted_sensor_values = AE(sensor_values.float())
            prvs_loss_for_AE_updated = criteria_AE(sensor_values.float(), predicted_sensor_values)
            AE.train()

        # calculating delta theta
        delta_theta = OrderedDict({k: inner_state[k] - final_state[k] for k in weights.keys()})

        # calculating phi gradient
        hnet_grads = torch.autograd.grad(list(weights.values()), hnet.parameters(), grad_outputs=list(delta_theta.values()))

        # update hnet weights
        for p, g in zip(hnet.parameters(), hnet_grads):
            p.grad = g
        torch.nn.utils.clip_grad_norm_(hnet.parameters(), 50)
        optimizer.step()

        # transform the server dataset using the user autoencoder
        # train a model on  transformed dataset in server side
        for i in range(params.inner_step_for_server):
            client_model[client_id].train()
            inner_optim.zero_grad()
            optimizer.zero_grad()
            batch = next(iter(server_loaders))
            sensor_values, activity = tuple(t.to(device) for t in batch)
            encoded_sensor_values = AE.encoder(sensor_values.float())
            predicted_activity = client_model[client_id](encoded_sensor_values)
            loss = criteria_model(predicted_activity, activity)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(client_model[client_id].parameters(), 50)
            inner_optim.step()
        optimizer.zero_grad()

        with torch.no_grad():
            client_model[client_id].eval()
            batch = next(iter(client_loader[client_id]))
            sensor_values, activity = tuple(t.to(device) for t in batch)
            encoded_sensor_values = AE.encoder(sensor_values.float())
            predicted_activity = client_model[client_id](encoded_sensor_values)
            prvs_loss_server_model = criteria_model(predicted_activity, activity)
            f1_score_server = f1_loss(activity, predicted_activity)
            client_model[client_id].train()

        # fine-tune the model on user labeled dataset (I commented that since looks like its not improving)
        for param in client_model[client_id].parameters():
            param.requires_grad = False

        client_model[client_id].fc2 = nn.Linear(params.model_hidden_layer, params.outputdim).to(device)

        for i in range(params.inner_step_for_client):
            inner_optim.zero_grad()
            optimizer.zero_grad()
            batch = next(iter(client_labeled_loaders[client_id]))
            sensor_values, activity = tuple(t.to(device) for t in batch)
            encoded_sensor_values = AE.encoder(sensor_values.float())
            predicted_activity = client_model[client_id](encoded_sensor_values)
            loss = criteria_model(predicted_activity, activity)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(client_model[client_id].parameters(), 50)
            inner_optim.step()
        optimizer.zero_grad()

        # Evaluate the model on user dataset
        with torch.no_grad():
            client_model[client_id].eval()
            batch = next(iter(client_loader[client_id]))
            sensor_values, activity = tuple(t.to(device) for t in batch)
            encoded_sensor_values = AE.encoder(sensor_values.float())
            predicted_activity = client_model[client_id](encoded_sensor_values)
            prvs_loss_fine_tuned = criteria_model(predicted_activity, activity)
            f1_score_user = f1_loss(activity, predicted_activity)
            client_model[client_id].train()
            step_iter.set_description(f"Step: {step+1}, Node ID: {client_id}, AE : {prvs_loss_for_AE:.4f}, AE fine: {prvs_loss_for_AE_updated:.4f}, Server loss: {prvs_loss_server_model:.4f}, server f1: {f1_score_server:.4f}, User loss: {prvs_loss_fine_tuned:.4f}, user f1: {f1_score_user:.4f}\n")

        # save results
        results['Step'].append(step+1)
        results['Node ID'].append(client_id)
        results['AE loss'].append(prvs_loss_for_AE.item())
        results['Server model loss'].append(prvs_loss_server_model.item())
        results['Client fine tuned loss'].append(prvs_loss_fine_tuned.item())
        results['AE fine tuned loss'].append(prvs_loss_for_AE_updated.item())
        results['Server f1'].append(f1_score_server.item())
        results['User f1'].append(f1_score_user.item())

    return results


if __name__ == '__main__':
    params = parameters()
    set_seed(params.seed)
    result = SemiPFL(params=params)
    pd.DataFrame.from_dict(result, orient="columns").to_csv("results.csv")
