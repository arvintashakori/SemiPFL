from collections import OrderedDict
import torch as t
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm

class HN(nn.Module):
    def __init__(
            self, n_nodes, embedding_dim, in_channels=3, out_dim=10, n_kernels=16, hidden_dim=100,
            spec_norm=False, n_hidden=1, n_kernels_enc=3, n_kernels_dec=2, latent_rep=4, stride_value=2, padding_value=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.n_kernels = n_kernels
        self.latent_rep = latent_rep
        self.embeddings = nn.Embedding(num_embeddings=n_nodes, embedding_dim=embedding_dim)
        layers = [
            spectral_norm(nn.Linear(embedding_dim, hidden_dim)) if spec_norm else nn.Linear(embedding_dim, hidden_dim),
        ]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim),
            )
        self.mlp = nn.Sequential(*layers)
        self.conv1_weights = nn.Linear(hidden_dim, self.n_kernels * self.in_channels * 3 * 3)
        self.conv1_bias = nn.Linear(hidden_dim, self.n_kernels)
        self.conv2_weights = nn.Linear(hidden_dim, self.latent_rep * self.n_kernels * 3 * 3)
        self.conv2_bias = nn.Linear(hidden_dim, self.latent_rep)
        self.t_conv1_weights = nn.Linear(hidden_dim, self.n_kernels * self.latent_rep * 2 * 2)
        self.t_conv1_bias = nn.Linear(hidden_dim, self.n_kernels)
        self.t_conv2_weights = nn.Linear(hidden_dim, self.in_channels * self.n_kernels * 2 * 2)
        self.t_conv2_bias = nn.Linear(hidden_dim, self.in_channels)
        if spec_norm:
            self.conv1_weights = spectral_norm(self.c1_weights)
            self.conv1_bias = spectral_norm(self.c1_bias)
            self.conv2_weights = spectral_norm(self.c2_weights)
            self.conv2_bias = spectral_norm(self.c2_bias)
            self.t_conv1_weights = spectral_norm(self.l1_weights)
            self.t_conv1_bias = spectral_norm(self.l1_bias)
            self.t_conv2_weights = spectral_norm(self.l2_weights)
            self.t_conv2_bias = spectral_norm(self.l2_bias)
    def forward(self, idx):
        emd = self.embeddings(idx)
        features = self.mlp(emd)
        weights = OrderedDict({
            "conv1.weight": self.conv1_weights(features).view(self.n_kernels, 3, 3, 3),
            "conv1.bias": self.conv1_bias(features).view(-1),
            "conv2.weight": self.conv2_weights(features).view(self.latent_rep, self.n_kernels, 3, 3),
            "conv2.bias": self.conv2_bias(features).view(-1),
            "t_conv1.weight": self.t_conv1_weights(features).view(self.latent_rep, self.n_kernels, 2, 2),
            "t_conv1.bias": self.t_conv1_bias(features).view(-1),
            "t_conv2.weight": self.t_conv2_weights(features).view(self.n_kernels, self.in_channels, 2, 2),
            "t_conv2.bias": self.t_conv2_bias(features).view(-1)
        })
        return weights


class Autoencoder(nn.Module):
    def __init__(self, input_size = 270, l1 = 128, l2 = 64, latent_rep = 32):
        super(Autoencoder, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(input_size, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, latent_rep)
        # Decoder
        self.fc4 = nn.Linear(latent_rep, l2)
        self.fc5 = nn.Linear(l2, l1)
        self.fc6 = nn.Linear(l1, input_size)

    def encoder(self, x): #update the model based on Wenwen
        x0 = x.view(x.size(0), -1)
        x1 = F.relu(self.fc1(x0))
        x2 = F.relu(self.fc2(x1))
        x3 = F.relu(self.fc3(x2))
        return x3

    def decoder(self, x): #update the model based on Wenwen
        x1 = F.relu(self.fc4(x))
        x2 = F.relu(self.fc5(x1))
        x3 = F.relu(self.fc6(x2))
        return x3

    def forward(self, input):
        latent_representation = self.encoder(input)
        output = self.decoder(latent_representation)
        return output

class BASEModel(nn.Module):
    def __init__(self, latent_rep = 32, l = 16, out_dim = 5, drop = 0.8):
        super(BASEModel, self).__init__()
        self.fc1 = nn.Linear(latent_rep, l)
        self.drop = nn.Dropout(p = drop)
        self.fc2 = nn.Linear(l, out_dim)

    def forward(self, x):
        x1 = F.relu(self.fc1(x0))
        x2 = self.drop(x1)
        x3 = self.fc2(x2)
        return x3
