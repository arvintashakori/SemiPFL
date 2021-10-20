from collections import OrderedDict
import torch as t
import torch.nn.functional as F
from torch import nn

t.manual_seed(0)


class HN(nn.Module):
    def __init__(self, n_nodes, embedding_dim, hidden_dim = 10, n_hidden = 1):
        super().__init__()

        self.embeddings = nn.Embedding(num_embeddings=n_nodes, embedding_dim=embedding_dim)
        layers = [nn.Linear(embedding_dim, hidden_dim),]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(hidden_dim, hidden_dim),)
        self.mlp = nn.Sequential(*layers)
        self.E1_weights = nn.Linear(hidden_dim, 256 * 270)
        self.E1_bias = nn.Linear(hidden_dim, 256 * 1)
        self.E2_weights = nn.Linear(hidden_dim, 128 * 256)
        self.E2_bias = nn.Linear(hidden_dim, 128 * 1)
        self.E3_weights = nn.Linear(hidden_dim, 64 * 128)
        self.E3_bias = nn.Linear(hidden_dim, 64 * 1)
        self.D1_weights = nn.Linear(hidden_dim, 128 * 64)
        self.D1_bias = nn.Linear(hidden_dim, 128 * 1)
        self.D2_weights = nn.Linear(hidden_dim, 256 * 128)
        self.D2_bias = nn.Linear(hidden_dim, 256 * 1)
        self.D3_weights = nn.Linear(hidden_dim, 270 * 256)
        self.D3_bias = nn.Linear(hidden_dim, 270 * 1)


    def forward(self, idx):
        emd = self.embeddings(idx)
        features = self.mlp(emd)
        weights = OrderedDict({
            "E1.weight": self.E1_weights(features).view(256, 270),
            "E1.bias": self.E1_bias(features).view(-1),
            "E2.weight": self.E2_weights(features).view(128, 256),
            "E2.bias": self.E2_bias(features).view(-1),
            "E3.weight": self.E3_weights(features).view(64, 128),
            "E3.bias": self.E3_bias(features).view(-1),
            "D1.weight": self.D1_weights(features).view(128, 64),
            "D1.bias": self.D1_bias(features).view(-1),
            "D2.weight": self.D2_weights(features).view(256, 128),
            "D2.bias": self.D2_bias(features).view(-1),
            "D3.weight": self.D3_weights(features).view(270, 256),
            "D3.bias": self.D3_bias(features).view(-1)
        })
        return weights


class Autoencoder(nn.Module):
    def __init__(self, inout_dim = 270, layer1 = 256, layer2 = 128, latent_rep = 64):
        super(Autoencoder, self).__init__()

        # Encoder
        self.E1 = nn.Linear(inout_dim, layer1)
        self.E2 = nn.Linear(layer1, layer2)
        self.E3 = nn.Linear(layer2, latent_rep)

        # Decoder
        self.D1 = nn.Linear(latent_rep, layer2)
        self.D2 = nn.Linear(layer2, layer1)
        self.D3 = nn.Linear(layer1, inout_dim)
        self.out = nn.Tanh()

    def encoder(self, x):
        x = F.relu(self.E1(x))
        x = F.relu(self.E2(x))
        x = self.E3(x)
        return x

    def decoder(self, x):
        x = F.relu(self.D1(x))
        x = F.relu(self.D2(x))
        x = self.out(self.D3(x))
        return x

    def forward(self, input):
        encoded_x = self.encoder(input)
        output = self.decoder(encoded_x)
        return output




class BASEModel(nn.Module):
    def __init__(self, latent_rep=64, out_dim=4, hidden_layer=16):
        super(BASEModel, self).__init__()

        self.batch1 = nn.BatchNorm1d(latent_rep)
        self.batch2 = nn.BatchNorm1d(hidden_layer)
        self.fc1 = nn.Linear(latent_rep, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, out_dim)

    def forward(self, x):
        x = self.batch1(x)
        x = self.batch2(F.relu(self.fc1(x)))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x
