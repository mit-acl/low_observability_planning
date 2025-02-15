'''
From: https://github.com/ethz-asl/cvae_exploration_planning?tab=readme-ov-file#Training-the-CVAE-Model
'''
import torch
import torch.nn.functional as F

############ CVAE definition ############

class Encoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Encoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, H)
        self.linear_mu = torch.nn.Linear(H, D_out)
        self.linear_logvar = torch.nn.Linear(H, D_out)
        # self.dropout = torch.nn.Dropout(p=0.5)
        self.H = H
        # print("Encoder shapes: ", D_in, H, D_out)

    # Encoder Q netowrk, approximate the latent feature with gaussian distribution,output mu and logvar
    def forward(self, x):
        # print("Encoder input shape: ", x.shape)
        x = F.relu(self.linear1(x))
        # x = self.dropout(x)
        x = F.relu(self.linear2(x))
        # x = self.dropout(x)
        x = F.relu(self.linear3(x))
        return self.linear_mu(x), self.linear_logvar(x)


class Decoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, H)
        self.linear4 = torch.nn.Linear(H, D_out)
        self.dropout = torch.nn.Dropout(p=0.2)
        # print("Decoder shapes: ", D_in, H, D_out)

    # Decoder P network, sampling from normal distribution and build the reconstruction
    def forward(self, x):
        # print("Decoder input shape: ", x.shape)
        x = F.relu(self.linear1(x))
        # x = self.dropout(x)
        x = F.relu(self.linear2(x))
        # x = self.dropout(x)
        x = F.relu(self.linear3(x))
        return self.linear4(x)


class CVAE(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(CVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    @staticmethod
    def _reparameterize(mu, logvar):
        eps = torch.randn_like(mu)
        return mu + torch.exp(logvar / 2) * eps

    def forward(self, state, cond):
        # print("State shape: ", state.shape)
        # print("Cond shape: ", cond.shape)
        x_in = torch.cat((state, cond), 1)
        # print("X_in shape: ", x_in.shape)
        mu, logvar = self.encoder(x_in)
        z = self._reparameterize(mu, logvar)
        z_in = torch.cat((z, cond), 1)
        return mu, logvar, self.decoder(z_in)



