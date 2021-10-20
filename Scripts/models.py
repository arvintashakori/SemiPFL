from collections import OrderedDict
import torch as t
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm

t.manual_seed(0)


class HN(nn.Module):
    def __init__(
            self, n_nodes, embedding_dim, ):
        super().__init__()
        self.embeddings = nn.Embedding(num_embeddings=n_nodes, embedding_dim=embedding_dim)
        layers = [spectral_norm(nn.Linear(embedding_dim, hidden_dim)) if spec_norm else nn.Linear(embedding_dim, hidden_dim),]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim),)
        self.mlp = nn.Sequential(*layers)
        self.conv1_weights = nn.Linear(
            hidden_dim, self.n_kernels * self.in_channels * self.n_kernels_enc * self.n_kernels_enc)
        self.conv1_bias = nn.Linear(hidden_dim, self.n_kernels)
        self.conv2_weights = nn.Linear(
            hidden_dim, self.latent_rep * self.n_kernels * self.n_kernels_enc * self.n_kernels_enc)
        self.conv2_bias = nn.Linear(hidden_dim, self.latent_rep)
        self.t_conv1_weights = nn.Linear(
            hidden_dim, self.n_kernels * self.latent_rep * self.n_kernels_dec * self.n_kernels_dec)
        self.t_conv1_bias = nn.Linear(hidden_dim, self.n_kernels)
        # self.t_conv2_weights = nn.Linear(
        #     hidden_dim, self.latent_rep * self.n_kernels * self.n_kernels_dec * self.n_kernels_dec)
        # self.t_conv2_bias = nn.Linear(hidden_dim, self.latent_rep)
        self.t_conv2_weights = nn.Linear(
            hidden_dim, self.in_channels * self.n_hidden * self.n_kernels_dec * 5)
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
            self.t_conv3_weights = spectral_norm(self.l3_weights)
            self.t_conv3_bias = spectral_norm(self.l3_bias)

    def forward(self, idx):
        emd = self.embeddings(idx)
        features = self.mlp(emd)
        weights = OrderedDict({
            "conv1.weight": self.conv1_weights(features).view(self.n_kernels, self.in_channels, self.n_kernels_enc,
                                                              self.n_kernels_enc),
            "conv1.bias": self.conv1_bias(features).view(-1),
            "conv2.weight": self.conv2_weights(features).view(self.latent_rep, self.n_kernels, self.n_kernels_enc,
                                                              self.n_kernels_enc),
            "conv2.bias": self.conv2_bias(features).view(-1),
            "t_conv1.weight": self.t_conv1_weights(features).view(self.latent_rep, self.n_hidden, self.n_kernels_dec,
                                                                  self.n_kernels_dec),
            "t_conv1.bias": self.t_conv1_bias(features).view(-1),
            # "t_conv2.weight": self.t_conv2_weights(features).view(self.n_hidden, self.latent_rep, self.n_kernels_dec,
            #                                                       self.n_kernels_dec),
            # "t_conv2.bias": self.t_conv2_bias(features).view(-1),
            "t_conv2.weight": self.t_conv2_weights(features).view(self.n_hidden, self.in_channels, self.n_kernels_dec,
                                                                  5),
            "t_conv2.bias": self.t_conv2_bias(features).view(-1)
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

    def encoder(self, x):
        x = F.relu(self.E1(x))
        x = F.relu(self.E2(x))
        x = self.E3(x)
        return x

    def decoder(self, x):
        x = F.relu(self.D1(x))
        x = F.relu(self.D2(x))
        x = F.tanh(self.D3(x))
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
