from models import HN, Autoencoder, BASEModel
from collections import OrderedDict, defaultdict
from node import Clients
import numpy as np
from tqdm import trange
import torch
import random
import torch.nn as nn
import torch.utils.data
from torchvision import transforms
torch.manual_seed(0)


class parameters:
    def __init__(self):
        self.labels_list = ['JOG', 'JUM',  'STD', 'WAL']
        self.data_address = './data/'
        self.trial_number = 0
        self.label_ratio = 0.20
        self.number_of_client = 58
        self.server_ID = 0
        self.batch_size = 128
        self.window_size = 30
        self.width = 9
        self.device = 'cpu'
        self.transform = 10
        self.total_number_of_clients = 59
        self.learning_rate = 1e-2
        self.steps = 5000
        self.inner_step_for_AE = 100
        self.inner_step_server_finetune = 100
        self.inner_step_for_server = 100
        self.inner_step_for_client = 100
        self.inner_lr = 1e-2
        self.inner_wd = 5e-5
        self.AEl1 = 128
        self.AEl2 = 64
        self.latent_rep = 32
        self.modell1 = 128
        self.drop = 0.8


def SemiPFL():
    #initialization
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    params = parameters()
    device = torch.device(params.device)

    #laoding data
    nodes = Clients(address=params.data_address,
                    trial_number=params.trial_number,
                    label_ratio=params.label_ratio,
                    #number_client=params.number_of_client,
                    server_ID=params.server_ID,
                    #batch_size=params.batch_size,
                    window_size=params.window_size,
                    width=params.width,
                    transform=transform,
                    num_user=params.total_number_of_clients)

    #model initialization
    hnet = HN(n_nodes=params.number_of_client, embedding_dim=int(
        1 + params.number_of_client / 4))  # initializing the hypernetwork
    AE = Autoencoder()  # initializing the autoencoder
    model = BASEModel()  # initilizing the base model

    #send models to device
    hnet.to(device)
    AE.to(device)
    model.to(device)

    #optimizer and loss functions
    optimizer = torch.optim.Adam(
        params=hnet.parameters(), lr=params.learning_rate)
    criteria_AE = torch.nn.BCEWithLogitsLoss()
    criteria_model = torch.nn.CrossEntropyLoss()  # Wenwen is using NLLLoss()

    #SemiPFL begins
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
            dataloader = torch.utils.data.DataLoader(
                nodes.client_loaders[client_id], batch_size=params.batch_size, shuffle=True)
            batch = next(iter(
                dataloader))
            sensor_values, _ = tuple(t.to(device) for t in batch)
            predicted_sensor_values = AE(sensor_values.float())
            prvs_loss = criteria_AE(predicted_sensor_values, sensor_values)
            AE.train()

        # inner updates -> obtaining theta_tilda
        dataloader = torch.utils.data.DataLoader(
            nodes.client_loaders[client_id], batch_size=params.batch_size, shuffle=True)
        for i in range(params.inner_step_server_finetune):
            AE.train()
            inner_optim.zero_grad()
            optimizer.zero_grad()
            batch = next(iter(dataloader))
            sensor_values, _ = tuple(t.to(device) for t in batch)
            predicted_sensor_values = AE(sensor_values.float())
            loss = criteria_AE(predicted_sensor_values, sensor_values)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(AE.parameters(), 50)
            inner_optim.step()
        optimizer.zero_grad()
        final_state = AE.state_dict()

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
        step_iter.set_description(
            f"Step: {step+1}, Node ID: {client_id}, Loss: {prvs_loss:.4f}")

        # transform the server dataset using the user autoencoder
        # train a model on  transformed dataset in server side
        dataloader = torch.utils.data.DataLoader(
            nodes.server_loaders, batch_size=params.batch_size, shuffle=True)
        for i in range(params.inner_step_for_server):
            user_model = model.train()
            inner_optim.zero_grad()
            optimizer.zero_grad()
            batch = next(iter(dataloader))
            sensor_values, activity = tuple(t.to(device) for t in batch)
            encoded_sensor_values = AE.encoder(sensor_values.float())
            predicted_activity = user_model(encoded_sensor_values)
            loss = criteria_model(predicted_activity, activity)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(user_model.parameters(), 50)
            inner_optim.step()
        optimizer.zero_grad()

        with torch.no_grad():
            dataloader = torch.utils.data.DataLoader(
                nodes.client_loaders[client_id], batch_size=params.batch_size, shuffle=True)
            user_model.eval()
            batch = next(iter(dataloader))
            sensor_values, activity = tuple(t.to(device) for t in batch)
            encoded_sensor_values = AE.encoder(sensor_values.float())
            predicted_activity = user_model(encoded_sensor_values)
            prvs_loss = criteria_model(predicted_activity, activity)
            user_model.train()
        print(prvs_loss)

        # fine-tune the model on user labeled dataset
        for param in user_model.parameters():
            param.requires_grad = False

        user_model.fc2 = nn.Linear(128, 20)

        dataloader = torch.utils.data.DataLoader(
            nodes.client_labeled_loaders[client_id], batch_size=params.batch_size, shuffle=True)
        for i in range(params.inner_step_for_client):
            inner_optim.zero_grad()
            optimizer.zero_grad()
            batch = next(iter(dataloader))
            sensor_values, activity = tuple(t.to(device) for t in batch)
            encoded_sensor_values = AE.encoder(sensor_values.float())
            predicted_activity = user_model(encoded_sensor_values)
            loss = criteria_model(predicted_activity, activity)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(user_model.parameters(), 50)
            inner_optim.step()
        optimizer.zero_grad()
        print(loss)

        # Evaluate the model on user dataset
        with torch.no_grad():
            dataloader = torch.utils.data.DataLoader(
                nodes.client_labeled_loaders[client_id], batch_size=params.batch_size, shuffle=True)
            user_model.eval()
            batch = next(iter(dataloader))
            sensor_values, activity = tuple(t.to(device) for t in batch)
            encoded_sensor_values = AE.encoder(sensor_values.float())
            predicted_activity = user_model(encoded_sensor_values)
            prvs_loss = criteria_model(predicted_activity, activity)
            user_model.train()
        print(prvs_loss)


SemiPFL()
