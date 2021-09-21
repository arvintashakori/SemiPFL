from models import HN, Autoencoder, BASEModel
import numpy as np
from tqdm import trange
import torch
import torch.nn as nn
import torch.utils.data
torch.manual_seed(0)

if __name__ == '__main__':
    #initialization
    device = get_device(gpus=args.gpu)
    #laoding data
    nodes = Dataloader(adress = address, num_users = num_users, labeled_ratio = labeled_ratio)
    #send models to device
    hnet = HN()
    AE = ConvAutoencoder()
    model = BASEModel()
    hnet.to(device)
    AE.to(device)
    model.to(device)
    #optimizer a and loss functions
    optimizer =  torch.optim.Adam(params=hnet.parameters(), lr=lr)
    criteria_AE = torch.nn.BCEWithLogitsLoss()
    criteria_model = torch.nn.CrossEntropyLoss() # Wenwen is using NLLLoss()
    #SemiPFL begins
    step_iter = trange(steps)
    results = defaultdict(list)
    for step in step_iter:
        hnet.train()
        # select client at random
        node_id = random.choice(range(num_nodes))
        # produce & load local network weights
        weights = hnet(torch.tensor([node_id], dtype=torch.long).to(device))
        AE.load_state_dict(weights)
        # init inner optimizer
        inner_optim = torch.optim.Adam(
            AE.parameters(), lr=inner_lr, weight_decay=inner_wd#, momentum=.9
        )
        # storing theta_i for later calculating delta theta
        inner_state = OrderedDict({k: tensor.data for k, tensor in weights.items()})
        # NOTE: evaluation on sent model
        with torch.no_grad():
            AE.eval()
            batch = next(iter(nodes.test_loaders[node_id]))
            img, label = tuple(t.to(device) for t in batch)
            pred = AE(img)
            prvs_loss = criteria_AE(pred, img)
            AE.train()
        # inner updates -> obtaining theta_tilda
        for i in range(inner_steps):
            AE.train()
            inner_optim.zero_grad()
            optimizer.zero_grad()
            batch = next(iter(nodes.train_loaders[node_id]))
            img, label = tuple(t.to(device) for t in batch)
            pred = AE(img)
            loss = criteria_AE(pred, img)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(AE.parameters(), 50)
            inner_optim.step()
