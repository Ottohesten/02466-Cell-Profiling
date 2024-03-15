import torch
from torch import nn


# base vae model architecture
# input img -> hidden -> mu, logvar -> reparameterization trick (sample point from distribution made from mu, logvar) -> decoder -> output img
class VAE(nn.Module):
    """
    Variational Autoencoder
    """
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim


        self.encoder = None
        self.mu = None
        self.sigma = None
        self.decoder = None

    def check_architecture(self):
        if self.encoder is None:
            raise NotImplementedError("Encoder not implemented")
        if self.mu is None:
            raise NotImplementedError("Mu not implemented")
        if self.sigma is None:
            raise NotImplementedError("Sigma not implemented")
        if self.decoder is None:
            raise NotImplementedError("Decoder not implemented")

    
    def encode(self, x):
        # q(z|x)
        h = self.encoder(x) # hidden
        mu = self.mu(h) # mean
        sigma = self.sigma(h) # log variance
        return mu, sigma # mean and log variance
    
    def decode(self, z):
        # p(x|z)
        return self.decoder(z)

    def reparameterize(self, mu, sigma):
        if self.training:
            eps = torch.randn_like(sigma)
            return mu + eps * sigma
        else:
            return mu

    def forward(self, x):
        mu, sigma = self.encode(x) 
        z = self.reparameterize(mu, sigma) # sample z from q(z|x) = mu + std * eps
        x_hat = self.decoder(z) # reconstruct x from z p(x|z)
        return x_hat, mu, sigma




# input img -> hidden -> mu, logvar -> reparameterization trick (sample point from distribution made from mu, logvar) -> decoder -> output img
class VAE_CNN(VAE):
    """
    Variational Autoencoder with CNN
    """
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__(input_dim, hidden_dim, latent_dim)

        # cnn encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1), # 28x28 -> 14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 14x14 -> 7x7
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # 7x7 -> 7x7
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128*7*7, hidden_dim),
            nn.ReLU()
        )
        # latent space
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.sigma = nn.Linear(hidden_dim, latent_dim)

        # self.decoder = nn.Sequential(
        #     nn.Linear(latent_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, input_dim),
        #     nn.Sigmoid()
        # )
        # make a decoder that returns a 28x28 image
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128*7*7),
            nn.ReLU(),
            nn.Unflatten(1, (128, 7, 7)),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1), # 7x7 -> 7x7
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # 7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1), # 14x14 -> 28x28
            nn.Sigmoid()
        )


class VAE_linear(VAE):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__(input_dim, hidden_dim, latent_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # latent space
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.sigma = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )




