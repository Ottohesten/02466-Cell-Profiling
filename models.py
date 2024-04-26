import torch
from torch import nn


# Autoencoder model architecture
class AutoEncoder(nn.Module):
    def __init__(self, latent_features=3):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.LeakyReLU(), # LeakyReLU prevents dead neurons by allowing a small gradient when the input is less than zero.
            nn.Dropout(0.1), # Prevent overfitting by turn off random neurons. Here we have chosen a relatively low percentage of 10%.
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 12),
            nn.LeakyReLU(),
            nn.Linear(12, latent_features)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_features, 12),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(12, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 28*28),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x





# base vae model architecture
# input img -> hidden -> mu, sigma -> reparameterization trick (sample point from distribution made from mu, sigma) -> decoder -> output img
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
        self.logvar = None
        self.decoder = None

    def check_architecture(self):
        if self.encoder is None:
            raise NotImplementedError("Encoder not implemented")
        if self.mu is None:
            raise NotImplementedError("Mu not implemented")
        if self.logvar is None:
            raise NotImplementedError("logvar not implemented")
        if self.decoder is None:
            raise NotImplementedError("Decoder not implemented")

    
    def encode(self, x): # posterior
        # q(z|x)
        h = self.encoder(x) # hidden
        mu = self.mu(h) # mean
        logvar = self.logvar(h) # log variance
        return mu, logvar # mean and log variance
    
    def decode(self, z):
        # p(x|z)
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return mu + eps*std
        else:
            return mu

    def forward(self, x):
        mu, logvar = self.encode(x) 
        z = self.reparameterize(mu, logvar) # sample z from q(z|x) = mu + std * eps
        x_hat = self.decoder(z) # reconstruct x from z p(x|z)
        return {'x_hat': x_hat, 'mu': mu, 'sigma': torch.exp(0.5*logvar), 'z': z}



# input img -> hidden -> mu, sigma -> reparameterization trick (sample point from distribution made from mu, sigma) -> decoder -> output img
class VAE_MNIST_CNN(VAE):
    """
    Variational Autoencoder for MNIST dataset with CNN architecture
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
            nn.Linear(128*7*7, 2*latent_dim),
            nn.ReLU()
        )
        # latent space
        self.mu = nn.Linear(latent_dim*2, latent_dim)
        self.logvar = nn.Linear(latent_dim*2, latent_dim)

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


class VAE_MNIST_linear(VAE):
    """
    Variational Autoencoder for the MNIST dataset with linear layers
    """
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__(input_dim, hidden_dim, latent_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2*latent_dim),
            nn.ReLU()
        )
        # latent space
        self.mu = nn.Linear(latent_dim*2, latent_dim)
        self.logvar = nn.Linear(latent_dim*2, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )


# input (68 x 68 img)
class VAE_CELL_linear(VAE):
    """
    Variational Autoencoder for the CELL dataset with linear layers
    """
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__(input_dim, hidden_dim, latent_dim)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(), # LeakyReLU prevents dead neurons by allowing a small gradient when the input is less than zero.
            nn.Linear(hidden_dim, 2*latent_dim),
            nn.LeakyReLU()
        )

        # latent space
        self.mu = nn.Linear(latent_dim*2, latent_dim)
        self.logvar = nn.Linear(latent_dim*2, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )


class VAE_CELL_CNN(VAE):
    """
    Variational Autoencoder for the CELL dataset with CNN architecture

    takes in 3x68x68 images

    Returns 3x68x68 images
    """
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__(input_dim, hidden_dim, latent_dim)

        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1), # 68x68 -> 34x34
            nn.LeakyReLU(), # LeakyReLU prevents dead neurons by allowing a small gradient when the input is less than zero.
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 34x34 -> 17x17
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 17x17 -> 9x9
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), # 9x9 -> 9x9
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(256*9*9, 2*latent_dim),
        )

        # latent space
        self.mu = nn.Linear(latent_dim*2, latent_dim)
        self.logvar = nn.Linear(latent_dim*2, latent_dim)

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 256*9*9),
            nn.LeakyReLU(),
            nn.Unflatten(1, (256, 9, 9)),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1), # 9x9 -> 9x9
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1), # 9x9 -> 17x17
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # 17x17 -> 34x34
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1), # 34x34 -> 68x68
            nn.Sigmoid()
        )

class CELL_CNN_CLASSIFIER(nn.Module):
    """
    Basic CNN classifier for the CELL dataset that just takes a 3x68x68 image and classifies it into one of the 13 classes
    """
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()


        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1), # 68x68 -> 34x34
            nn.LeakyReLU(), # LeakyReLU prevents dead neurons by allowing a small gradient when the input is less than zero.
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 34x34 -> 17x17
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 17x17 -> 9x9
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), # 9x9 -> 9x9
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(256*9*9, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

class VAE_LAFARGE(VAE):
    def __init__(self,input_dim, hidden_dim, latent_dim=256):
        super().__init__(input_dim, hidden_dim, latent_dim)
        
        # encoder with max pooling and batch normalization
        self.encoder = nn.Sequential(
            # block 1
            nn.Conv2d(3, 32, kernel_size=5), # 68x68x3 -> 64x64x32
            nn.LeakyReLU(),
            nn.MaxPool2d(2), # 64x64x32 -> 32x32x32
            nn.BatchNorm2d(32),

            # block 2
            nn.Conv2d(32, 64, kernel_size=5), # 32x32x32 -> 28x28x64
            nn.LeakyReLU(),
            nn.MaxPool2d(2), # 28x28x64 -> 14x14x64
            nn.BatchNorm2d(64),

            # block 3
            nn.Conv2d(64, 128, kernel_size=5), # 14x14x64 -> 10x10x128
            nn.LeakyReLU(),
            nn.MaxPool2d(2), # 10x10x128 -> 5x5x128
            nn.BatchNorm2d(128),

            # block 4
            nn.Conv2d(128, 256, kernel_size=5), # 5x5x128 -> 1x1x256
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.Flatten(),
            nn.Linear(256, latent_dim*2), # 1x1x256 -> 1x1x512
            # nn.Sigmoid()
        )

        self.mu = nn.Linear(latent_dim*2, latent_dim)
        self.logvar = nn.Linear(latent_dim*2, latent_dim)

        # decoder with upsampling and batch normalization turn to 68x68x3 in 4 blocks
        self.decoder = nn.Sequential(
            # block 1
            nn.Unflatten(1, (latent_dim, 1, 1)),
            nn.ConvTranspose2d(latent_dim, 128, kernel_size=5), # 1x1x256 -> 5x5x128
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2), # 5x5x128 -> 10x10x128

            # block 2
            nn.ConvTranspose2d(128, 64, kernel_size=5), # 10x10x128 -> 14x14x64
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Upsample(scale_factor=2), # 14x14x64 -> 28x28x64

            # block 3
            nn.ConvTranspose2d(64, 32, kernel_size=5), # 28x28x64 -> 32x32x32
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.Upsample(scale_factor=2), # 32x32x32 -> 64x64x32
            
            # block 4
            nn.ConvTranspose2d(32, 3, kernel_size=5), # 64x64x32 -> 68x68x3
            # nn.BatchNorm2d(3),
            nn.Sigmoid()
            
        )        




import torch.nn.functional as F # for the activation functions
class VAE_CELL_CNN_CLASSIFIER(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_classes):
        super(VAE_CELL_CNN_CLASSIFIER, self).__init__()
        
        # Encoder part
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1), # 68x68 -> 34x34
            nn.LeakyReLU(), # LeakyReLU prevents dead neurons by allowing a small gradient when the input is less than zero.
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 34x34 -> 17x17
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 17x17 -> 9x9
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), # 9x9 -> 9x9
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(256*9*9, hidden_dim),
            nn.Sigmoid()
        )
        
        # Latent space
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder part
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 256*9*9),
            nn.LeakyReLU(),
            nn.Unflatten(1, (256, 9, 9)),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1), # 9x9 -> 9x9
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1), # 9x9 -> 17x17
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # 17x17 -> 34x34
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1), # 34x34 -> 68x68
            nn.Sigmoid() # Last layer is a sigmoid function to ensure the output is between 0 and 1
        )

        # Classification head
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z)
        class_logits = self.classifier(encoded)
        return x_hat, mu, log_var, class_logits

    def forward_classifier(self, x):
        encoded = self.encoder(x)
        class_logits = self.classifier(encoded)
        return class_logits

