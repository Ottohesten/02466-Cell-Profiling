import torch
import torch.nn.functional as F

def loss_function(x, x_hat, mu, sigma, beta=1.0):
    MSE = F.mse_loss(x_hat, x, reduction='sum')
    # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) # kullback-leibler divergence between q(z|x) and p(z)
    KLD = -0.5 * torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
    # return {"BCE": BCE, "KLD": KLD, "loss": BCE + beta * KLD}
    return {"MSE": MSE, "KLD": KLD, "loss": MSE + beta * KLD}


def loss_function_logvar(x, x_hat, mu, logvar, beta=1.0):
    # MSE
    MSE = F.mse_loss(x_hat, x, reduction='sum')

    # KLD
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return {"MSE": MSE, "KLD": KLD, "loss": MSE + beta * KLD}