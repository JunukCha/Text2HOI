import torch
from torch import Tensor
import torch.nn as nn

from lib.networks.pointnet import PointNetLatent


class CVAE_BASE(nn.Module):
    def __init__(
        self, 
        **kwargs,
    ):
        super().__init__()
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def get_mean_logvar(self, encoded, dim):
        mu, logvar = torch.chunk(encoded, 2, dim=dim)
        return mu, logvar
    
    
class SeqCVAE(CVAE_BASE): # Contact CVAE
    def __init__(
        self, 
        latent_dim=512, 
        cond_dim=512, 
        encoder_layer_dims=[1, 512], 
        decoder_layer_dims=[512, 1], 
        final_sigmoid=True, 
        **kwargs,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(
            layer_dims=encoder_layer_dims, 
            latent_dim=latent_dim, 
            cond_dim=cond_dim, 
        )

        self.decoder = Decoder(
            layer_dims=decoder_layer_dims, 
            latent_dim=latent_dim, 
            cond_dim=cond_dim, 
            final_sigmoid=final_sigmoid,
        )
        
    def forward(self, x, condition):
        encoded = self.encoder(torch.cat([x, condition], dim=1))
        mu, logvar = self.get_mean_logvar(encoded, dim=1)
        
        z = self.reparameterize(mu, logvar)
        decoded = self.decoder(torch.cat([z, condition], dim=1))
        return decoded, mu, logvar
    
    def decode(self, condition):
        bs = condition.shape[0]
        z = torch.randn((bs, self.latent_dim), device=condition.device)
        seq_length = self.decoder(torch.cat([z, condition], dim=1))
        return seq_length
        
        
class CTCVAE(CVAE_BASE): # Contact CVAE
    def __init__(
        self, 
        latent_dim=64, 
        cond_dim=1601, 
        decoder_layer_dims=[512, 256, 128, 1], 
        final_sigmoid=True, 
        in_dim=4, 
        **kwargs
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = PointNetLatent(
            self.latent_dim*2, 
            feature_transform=False, 
            in_dim=in_dim, 
        )
        
        self.decoder = Decoder(
            layer_dims=decoder_layer_dims, 
            latent_dim=latent_dim, 
            cond_dim=cond_dim, 
            final_sigmoid=final_sigmoid, 
        )

    def forward(self, x_map, cond_map):
        encoded = self.encoder(x_map)
        mu, logvar = self.get_mean_logvar(encoded, dim=1)
        
        z_map = self.reparameterize(mu, logvar)
        z_map = z_map.unsqueeze(1)
        z_map = z_map.expand(-1, cond_map.shape[1], -1)

        contact_map = self.decoder(
            torch.cat([z_map, cond_map], dim=2)
        )
        return contact_map, mu, logvar

    def decode(self, cond_map):
        bs = cond_map.shape[0]
        z_map = torch.randn((bs, 1, self.latent_dim), device=cond_map.device)
        z_map = z_map.expand(-1, cond_map.shape[1], -1)
        contact_map = self.decoder(
            torch.cat([z_map, cond_map], dim=2)
        )
        return contact_map
    
    
class Encoder(nn.Module):
    def __init__(self, layer_dims, latent_dim, cond_dim):
        super().__init__()
        self.MLP = nn.Sequential()
        layer_dims[0] += cond_dim

        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
            self.MLP.append(nn.Linear(in_dim, out_dim))
            self.MLP.append(nn.ReLU())

        self.linear_mean_log = nn.Linear(layer_dims[-1], latent_dim*2)
    
    def forward(self, x):
        x = self.MLP(x)
        encoded = self.linear_mean_log(x)
        return encoded


class Decoder(nn.Module):
    def __init__(
        self, 
        layer_dims, 
        latent_dim, 
        cond_dim, 
        final_sigmoid=False
    ):
        super().__init__()
        self.final_sigmoid = final_sigmoid
        self.MLP = nn.Sequential()
        input_size = latent_dim + cond_dim

        for i, (in_dim, out_dim) in enumerate(zip([input_size]+layer_dims[:-2], layer_dims[:-1])):
            self.MLP.append(
                nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.LeakyReLU(negative_slope=0.2),
                )
            )
        self.MLP.append(
            nn.Sequential(
                nn.Linear(layer_dims[-2], layer_dims[-1]),
            )
        )

        if self.final_sigmoid:
            self.MLP.append(nn.Sigmoid())
    
    def forward(self, x):
        x = self.MLP(x)
        return x