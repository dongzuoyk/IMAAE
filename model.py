import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class Encoder(nn.Module):
    def __init__(self,data_size,latent_dim):
        super(Encoder,self).__init__()
        
        self.relu = nn.ReLU(inplace=True)
        
        self.enc = nn.Sequential(
            nn.Linear(data_size, data_size//2),
            nn.BatchNorm1d(data_size//2),
            Mish(),
            nn.Linear(data_size//2, data_size//4),
            nn.BatchNorm1d(data_size//4),
            Mish(),
            nn.Linear(data_size//4, latent_dim),
            nn.BatchNorm1d(latent_dim),
            Mish()
        )
        
        self.mu_enc = nn.Linear(latent_dim, latent_dim)
        self.var_enc = nn.Linear(latent_dim, latent_dim)
        
    def reparameterize(self, mu, var):
        return Normal(mu, var.sqrt()).rsample()
        
    def forward(self,x):
        
        q = self.enc(x)
        mu = self.mu_enc(q)
        var = torch.exp(self.var_enc(q))
        
        z = self.reparameterize(mu, var)
        
        return z, mu, var
    
    
class Decoder(nn.Module):
    def __init__(self,data_size,latent_dim):
        super(Decoder,self).__init__()
        
        self.relu = nn.ReLU(inplace=True)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, data_size//4),
            nn.BatchNorm1d(data_size//4),
            Mish(),
            nn.Linear(data_size//4, data_size//2),
            nn.BatchNorm1d(data_size//2),
            Mish(),
            nn.Linear(data_size//2, data_size),
        )
        
        

    def forward(self, latent_data):
        
        gen_data = self.decoder(latent_data)

        return self.relu(gen_data)


    
class Discriminator(nn.Module):
    def __init__(self, data_size):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(data_size, data_size//2),
            Mish(),
            nn.Linear(data_size//2, data_size//4),
            Mish(),
            nn.Linear(data_size//4, data_size//8),
            Mish(),
        )

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(data_size//8, 1))

    def forward(self, data):
        out = self.model(data)
        validity = self.adv_layer(out)
        return validity
    
class Discriminator2(nn.Module):
    def __init__(self,data_size):
        super(Discriminator2, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(data_size, 1024),
            Mish(),
            nn.Linear(1024, 512),
            Mish(),
            nn.Linear(512, 256),
            Mish(),
        )

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(256, 1))

    def forward(self, data):
        out = self.model(data)
        validity = self.adv_layer(out)
        return validity
