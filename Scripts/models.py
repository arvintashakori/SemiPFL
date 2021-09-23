from collections import OrderedDict
import torch as t
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm

class CNNHyper(nn.Module):
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


class ConvAutoencoder(nn.Module):
    def __init__(self, inout_channels=3, hidden=16, n_kernels_enc=3,
     n_kernels_dec=2, latent_rep=4, stride_value=2, padding_value=1):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(inout_channels, hidden, n_kernels_enc, padding=padding_value)
        self.conv2 = nn.Conv2d(hidden, latent_rep, n_kernels_enc, padding=padding_value)
        self.pool = nn.MaxPool2d(2, 2)
        # Decoder
        self.t_conv1 = nn.ConvTranspose2d(latent_rep, hidden, n_kernels_dec, stride=stride_value)
        self.t_conv2 = nn.ConvTranspose2d(hidden, inout_channels, n_kernels_dec, stride=stride_value)
    def encoder(self, x):
        z = self.pool(F.relu(self.conv1(x)))
        z = self.pool(F.relu(self.conv2(z)))
        return z
    def decoder(self, x):
        z = F.relu(self.t_conv1(x))
        z = t.sigmoid(self.t_conv2(z))
        return z
    def forward(self, input):
        latent_representation = self.encoder(input)
        output = self.decoder(latent_representation)
        return output

class BASEModel(nn.Module):
    def __init__(self, latent_rep=256, out_dim=10):
        super(BASEModel, self).__init__()
        self.fc1 = nn.Linear(latent_rep, 32)
        self.drop = nn.Dropout(p=0.8)
        self.fc2 = nn.Linear(32, out_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x