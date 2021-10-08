from collections import OrderedDict
import torch as t
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm

t.manual_seed(0)


class HN(nn.Module):
    def __init__(
            self, n_nodes, embedding_dim, in_channels=1, out_dim=4, n_kernels=16, hidden_dim=100,
            spec_norm=False, n_hidden=1, n_kernels_enc=3, n_kernels_dec=3, latent_rep=4, stride_value=1,
            padding_value=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.n_kernels = n_kernels
        self.n_kernels_enc = n_kernels_enc
        self.n_kernels_dec = n_kernels_dec
        self.latent_rep = latent_rep
        self.n_hidden = n_hidden
        self.embeddings = nn.Embedding(
            num_embeddings=n_nodes, embedding_dim=embedding_dim)
        layers = [
            spectral_norm(nn.Linear(embedding_dim, hidden_dim)) if spec_norm else nn.Linear(
                embedding_dim, hidden_dim),
        ]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(
                    hidden_dim, hidden_dim),
            )
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
        self.t_conv2_weights = nn.Linear(
            hidden_dim, self.latent_rep * self.n_kernels * self.n_kernels_dec * self.n_kernels_dec)
        self.t_conv2_bias = nn.Linear(hidden_dim, self.latent_rep)
        self.t_conv3_weights = nn.Linear(
            hidden_dim, self.in_channels * self.latent_rep * self.n_kernels_dec * 5)
        self.t_conv3_bias = nn.Linear(hidden_dim, self.in_channels)
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
            "t_conv2.weight": self.t_conv2_weights(features).view(self.n_hidden, self.latent_rep, self.n_kernels_dec,
                                                                  self.n_kernels_dec),
            "t_conv2.bias": self.t_conv2_bias(features).view(-1),
            "t_conv3.weight": self.t_conv3_weights(features).view(self.latent_rep, self.in_channels, self.n_kernels_dec,
                                                                  5),
            "t_conv3.bias": self.t_conv3_bias(features).view(-1)
        })
        return weights


class Autoencoder(nn.Module):
    def __init__(self, inout_channels=1, hidden=16, n_kernels_enc=3,
                 n_kernels_dec=3, latent_rep=4, stride_value=1, padding_value=1):
        super(Autoencoder, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(inout_channels, hidden,
                               n_kernels_enc, padding=padding_value)  # hidden*9*30
        self.conv2 = nn.Conv2d(
            hidden, latent_rep, n_kernels_enc, padding=padding_value)  # latent_rep*9*30
        self.pool = nn.MaxPool2d((3, 5), stride=(3, 5))  # latent_rep*3*6
        # self.pool = nn.MaxPool2d(3, stride=2)
        # self.conv3 = nn.Conv2d(8, latent_rep, n_kernels_enc, padding=padding_value)
        # batch norms
        # self.batchNorm1 = nn.BatchNorm2d(16)
        # self.batchNorm2 = nn.BatchNorm2d(4)
        # Decoder

        #        self.t_conv1 = nn.ConvTranspose2d(
        #            latent_rep, hidden, kernel_size=(n_kernels_dec, n_kernels_dec), stride=stride_value, padding=padding_value)
        #        self.t_conv2 = nn.ConvTranspose2d(
        #            hidden, inout_channels, kernel_size=(n_kernels_dec, n_kernels_dec), stride=stride_value,
        #            padding=padding_value)

        self.t_conv1 = nn.ConvTranspose2d(
            latent_rep, hidden, kernel_size=(n_kernels_dec, n_kernels_dec), stride=stride_value, padding=padding_value)
        self.t_conv2 = nn.ConvTranspose2d(
            hidden, latent_rep, kernel_size=(n_kernels_dec, n_kernels_dec), stride=stride_value, padding=padding_value)
        self.t_conv3 = nn.ConvTranspose2d(
            latent_rep, inout_channels, kernel_size=(3, 5), stride=(3, 5), padding=0)
        print('')

    def encoder(self, x):
        z = F.relu(self.conv1(x))
        z = self.pool(F.relu(self.conv2(z)))
        # z = F.relu(self.conv3(z))
        return z

    def decoder(self, x):
        z = F.relu(self.t_conv1(x))
        z = F.relu(self.t_conv2(z))
        z = F.relu(self.t_conv3(z))
        # z = t.sigmoid(self.t_conv3(z))
        return z

    def forward(self, input):
        latent_representation = self.encoder(input)
        output = self.decoder(latent_representation)
        return output

    def forward(self, input):
        latent_representation = self.encoder(input)
        output = self.decoder(latent_representation)
        return output


class BASEModel(nn.Module):
    def __init__(self, latent_rep=4 * 3 * 6, out_dim=4, hidden_layer=128):  # 9 * 30 * 4
        super(BASEModel, self).__init__()
        self.fc1 = nn.Linear(latent_rep, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, out_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x
