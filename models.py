import torch
from torch import nn


# Autoencoder model architecture
class AutoEncoder(nn.Module):
    def __init__(self, latent_dim=3):
        super(AutoEncoder, self).__init__()
        self.latent_dim = latent_dim

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
            nn.Linear(12, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 12),
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


class CELL_CNN_AutoEncoder(nn.Module):
    """
    Convolutional Autoencoder for the CELL dataset
    """
    def __init__(self, latent_dim=256):
        super().__init__()
        self.latent_dim = latent_dim

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
            nn.Linear(256, latent_dim)
        )

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(),
            nn.Unflatten(1, (256, 1, 1)),
            nn.ConvTranspose2d(256, 128, kernel_size=5), # 1x1x256 -> 5x5x128
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2), # 5x5x128 -> 10x10x128

            nn.ConvTranspose2d(128, 64, kernel_size=5), # 10x10x128 -> 14x14x64
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Upsample(scale_factor=2), # 14x14x64 -> 28x28x64

            nn.ConvTranspose2d(64, 32, kernel_size=5), # 28x28x64 -> 32x32x32
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.Upsample(scale_factor=2), # 32x32x32 -> 64x64x32
            
            nn.ConvTranspose2d(32, 3, kernel_size=5), # 64x64x32 -> 68x68x3
            nn.BatchNorm2d(3),
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
        # self.mu = None
        # self.logvar = None
        self.decoder = None

    def check_architecture(self):
        if self.encoder is None:
            raise NotImplementedError("Encoder not implemented")
        # if self.mu is None:
        #     raise NotImplementedError("Mu not implemented")
        # if self.logvar is None:
        #     raise NotImplementedError("logvar not implemented")
        if self.decoder is None:
            raise NotImplementedError("Decoder not implemented")

    
    def encode(self, x): # posterior
        # q(z|x)
        h = self.encoder(x) # hidden
        mu, logvar = torch.chunk(h, 2, dim=1) # split the hidden layer into two parts
        return mu, logvar # mean and log variance
    
    def decode(self, z):
        # p(x|z)
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        # if self.training:
        std = torch.exp(0.5*logvar)     # Compute sigma from log(sigma^2)/logvar.
        eps = torch.randn_like(std)     # Sample epsilon from a standard normal distribution.
        return mu + eps*std             # Reparameterization trick.
        # else:
        #     return mu

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
        # # latent space
        # self.mu = nn.Linear(latent_dim*2, latent_dim)
        # self.logvar = nn.Linear(latent_dim*2, latent_dim)

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
        # # latent space
        # self.mu = nn.Linear(latent_dim*2, latent_dim)
        # self.logvar = nn.Linear(latent_dim*2, latent_dim)

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

        # # latent space
        # self.mu = nn.Linear(latent_dim*2, latent_dim)
        # self.logvar = nn.Linear(latent_dim*2, latent_dim)

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

        # # latent space
        # self.mu = nn.Linear(latent_dim*2, latent_dim)
        # self.logvar = nn.Linear(latent_dim*2, latent_dim)

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

# even bigger vae cnn model
class VAE_CELL_CNN_2(VAE):
    """
    Variational Autoencoder for the CELL dataset with CNN architecture

    takes in 3x68x68 images

    Returns 3x68x68 images
    """
    def __init__(self, input_dim=(3, 68, 68), hidden_dim=1024, latent_dim=512):
        super().__init__(input_dim, hidden_dim, latent_dim)


                # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1), # 3x68x68 -> 128x68x68
            nn.LeakyReLU(), # LeakyReLU prevents dead neurons by allowing a small gradient when the input is less than zero.
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # 128x68x68 -> 256x34x34
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), # 256x34x34 -> 512x17x17
            nn.LeakyReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1), # 512x17x17 -> 1024x9x9
            nn.LeakyReLU(),
            nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1), # 1024x9x9 -> 2048x5x5
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(2048*5*5, 2*latent_dim),
        )

        # # latent space
        # self.mu = nn.Linear(latent_dim*2, latent_dim)
        # self.logvar = nn.Linear(latent_dim*2, latent_dim)

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 2048*5*5),
            nn.LeakyReLU(),
            nn.Unflatten(1, (2048, 5, 5)),
            nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1), # 5x5 -> 9x9
            nn.LeakyReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1), # 9x9 -> 17x17
            nn.LeakyReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1), # 17x17 -> 34x34
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), # 34x34 -> 68x68
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 3, kernel_size=3, stride=1, padding=1), # 68x68 -> 68x68
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

# cnn classifier with 3 blocks of convolutions
class CELL_CNN_CLASSIFIER_2(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()

        self.net = nn.Sequential(
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

            nn.Flatten(),
            nn.Linear(128*5*5, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# latent space classifier
class LatentClassifier(nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_classes):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)
    

class LatentClassifier_2(nn.Module):
    """
    Classify latent sample with dimensions 2
    """
    def __init__(self, latent_dim, hidden_dim, num_classes):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class VAE_LAFARGE(VAE):
    def __init__(self,input_dim, hidden_dim, latent_dim=256):
        super().__init__(input_dim, hidden_dim, latent_dim)
        
        # encoder with max pooling and batch normalization
        self.encoder = nn.Sequential(
            # block 1
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=0), # 68x68x3 -> 64x64x32
            nn.LeakyReLU(),
            nn.MaxPool2d(2), # 64x64x32 -> 32x32x32
            nn.BatchNorm2d(32),

            # block 2
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=0), # 32x32x32 -> 28x28x64
            nn.LeakyReLU(),
            nn.MaxPool2d(2), # 28x28x64 -> 14x14x64
            nn.BatchNorm2d(64),

            # block 3
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=0), # 14x14x64 -> 10x10x128
            nn.LeakyReLU(),
            nn.MaxPool2d(2), # 10x10x128 -> 5x5x128
            nn.BatchNorm2d(128),

            # block 4
            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=0), # 5x5x128 -> 1x1x256
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.Flatten(),
            nn.Linear(256, latent_dim*2), # 1x1x256 -> 1x1x512
            # nn.Sigmoid()
        )

        # self.mu = nn.Linear(latent_dim*2, latent_dim)
        # self.logvar = nn.Linear(latent_dim*2, latent_dim)

        # decoder with upsampling and batch normalization turn to 68x68x3 in 4 blocks
        self.decoder = nn.Sequential(
            # block 1
            nn.Unflatten(1, (latent_dim, 1, 1)),
            nn.ConvTranspose2d(latent_dim, 128, kernel_size=5, stride=1, padding=0), # 1x1x256 -> 5x5x128
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2), # 5x5x128 -> 10x10x128

            # block 2
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=1, padding=0), # 10x10x128 -> 14x14x64
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Upsample(scale_factor=2), # 14x14x64 -> 28x28x64

            # block 3
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=1, padding=0), # 28x28x64 -> 32x32x32
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.Upsample(scale_factor=2), # 32x32x32 -> 64x64x32
            
            # block 4
            nn.ConvTranspose2d(32, 3, kernel_size=5, stride=1, padding=0), # 64x64x32 -> 68x68x3
            # nn.BatchNorm2d(3),
            nn.Sigmoid()
            
        )        

class VAE_LAFARGE_v2(VAE):
    def __init__(self, input_dim, hidden_dim, latent_dim=256):
        super().__init__(input_dim, hidden_dim, latent_dim)
        
        # Encoder with max pooling, batch normalization, and dropout for regularization
        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=6, stride=1, padding=2, dilation=2),  # 68x68x3 -> 62x62x32
            nn.LeakyReLU(),
            nn.MaxPool2d(2),  # 62x62x32 -> 31x31x32
            nn.BatchNorm2d(32),
            nn.Dropout(0.1), # Dropout layer to prevent overfitting. Dropout rate of 10%.

            # # Block 2
            nn.Conv2d(32, 64, kernel_size=6, stride=1, padding=2, dilation=2),  # 31x31x32 -> 25x25x64
            nn.LeakyReLU(),
            nn.MaxPool2d(2),  # 25x25x64 -> 12x12x64
            nn.BatchNorm2d(64),
            nn.Dropout(0.1),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=6, stride=1, padding=2, dilation=2),  # 12x12x64 -> 6x6x128
            nn.LeakyReLU(),
            nn.MaxPool2d(2),  # 6x6x128 -> 3x3x128
            nn.BatchNorm2d(128),
            nn.Dropout(0.1),

            # # Block 4
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),  # 3x3x128 -> 1x1x256
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.Flatten(),
            nn.Linear(1*1*256, latent_dim*2),  # 1x1x256 -> latent_dim*2
        )

        # Decoder with upsampling, batch normalization, and larger kernel sizes to increase receptive field
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (latent_dim, 1, 1)),
            nn.ConvTranspose2d(latent_dim, 128, kernel_size=3, stride=1, padding=0),  # 1x1x256 -> 3x3x128
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),  # 3x3x128 -> 6x6x128
            nn.Dropout(0.1),

            nn.ConvTranspose2d(128, 64, kernel_size=6, stride=1, padding=2, dilation=2),  # 6x6x128 -> 12x12x64
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Upsample(scale_factor=2),  # 12x12x64 -> 24x24x64
            nn.Dropout(0.1),

            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=1, padding=2, dilation=2, output_padding=1),  # 24x24x64 -> 31x31x32
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.Upsample(scale_factor=2),  # 31x31x32 -> 62x62x32
            nn.Dropout(0.1),
            
            nn.ConvTranspose2d(32, 3, kernel_size=6, stride=1, padding=2, dilation=2),  # 64x64x32 -> 68x68x3
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
        std = torch.exp(0.5*log_var)        # Compute sigma from log(sigma^2)/log_var.
        eps = torch.randn_like(std)         # Sample epsilon from a standard normal distribution.
        return mu + eps*std                 # Reparameterization trick.

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



if __name__ == "__main__":
    # test the number of trainable parameters

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    vae = VAE_CELL_CNN_2((3, 68, 68), 512, 256)
    print(f"Number of trainable parameters in VAE_CELL_CNN_2: {count_parameters(vae):,}")
    