import torch
import torch.nn.functional as F

def loss_function(x, x_hat, mu, sigma, beta=1.0):
    MSE = F.mse_loss(x_hat, x, reduction='sum')
    # MSE = F.mse_loss(x_hat, x, reduction='none').sum(dim=(1, 2, 3)).mean()
    # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) # kullback-leibler divergence between q(z|x) and p(z)
    KLD = -0.5 * torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
    # return {"BCE": BCE, "KLD": KLD, "loss": BCE + beta * KLD}
    return {"MSE": MSE, "KLD": KLD, "loss": MSE + beta * KLD}



# def loss_function_logvar(x, x_hat, mu, logvar, beta=1.0):
#     # MSE
#     MSE = F.mse_loss(x_hat, x, reduction='sum')

#     # KLD
#     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

#     return {"MSE": MSE, "KLD": KLD, "loss": MSE + beta * KLD}


def loss_function_mean(x, x_hat, mu, sigma, beta=1.0):
    batch_size = x.size(0)
    MSE = F.mse_loss(x_hat, x, reduction='sum') / batch_size
    # MSE = F.mse_loss(x_hat, x, reduction='none').sum(dim=(1, 2, 3)).mean()
    # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) # kullback-leibler divergence between q(z|x) and p(z)
    KLD = -0.5 * torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2)) / batch_size
    # return {"BCE": BCE, "KLD": KLD, "loss": BCE + beta * KLD}
    return {"MSE": MSE, "KLD": KLD, "loss": MSE + beta * KLD}


def loss_function_mean_2(x, x_hat, mu, sigma, beta=1.0):
    batch_size = x.size(0)
    MSE = F.mse_loss(x_hat, x, reduction='none').sum(dim=(1, 2, 3)).mean()
    # MSE = F.mse_loss(x_hat, x, reduction='none').sum(dim=(1, 2, 3)).mean()
    # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) # kullback-leibler divergence between q(z|x) and p(z)
    KLD = -0.5 * torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2)) / batch_size
    # return {"BCE": BCE, "KLD": KLD, "loss": BCE + beta * KLD}
    return {"MSE": MSE, "KLD": KLD, "loss": MSE + beta * KLD}

# use torch.distributions
def loss_function_elbo(x, x_hat, mu, sigma, beta=1.0):
    """
    loss function using torch.distributions
    """
    from torch.distributions import Normal, kl_divergence
    # MSE
    MSE = F.mse_loss(x_hat, x, reduction='sum')
    # KLD
    q_z_x = Normal(mu, sigma)
    p_z = Normal(0, 1)
    KLD = kl_divergence(q_z_x, p_z).sum()
    # return {"MSE": MSE, "KLD": KLD, "loss": MSE + beta * KLD}
    return {"MSE": MSE, "KLD": KLD, "loss": MSE + beta * KLD}


def loss_function_elbo_2(x, x_hat, mu, sigma, beta=1.0):
    # Calculate reconstruction loss (negative log likelihood of the input under the output distribution)
    from torch.distributions import Normal
    MSE = F.mse_loss(x_hat, x, reduction='sum')

    # Calculate KL divergence loss
    q_z_x = Normal(mu, sigma)
    p_z = Normal(torch.zeros_like(mu), torch.ones_like(sigma))
    kl_loss = torch.distributions.kl.kl_divergence(q_z_x, p_z).sum()  # Summing over all elements

    # Total ELBO loss
    loss = MSE + beta * kl_loss

    return {"MSE": MSE, "KLD": kl_loss, "loss": loss}

def reduce(x):
    return torch.mean(x)

def loss_function_elbo_3(x, x_hat, mu, sigma, beta=1.0):
    log_px = F.mse_loss(x_hat, x, reduction='none')
    log_px = reduce(log_px)

    kl_div = -0.5 * torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
    kl_div = reduce(kl_div)

    loss = log_px + beta * kl_div

    return {"MSE": log_px, "KLD": kl_div, "loss": loss}


def loss_function_elbo_4(x, x_hat, mu, sigma, beta=1.0):
    """
    use normal distribution and log_prob
    """
    from torch.distributions import Normal

    px = Normal(x_hat, sigma)
    log_px = px.log_prob(x).sum()

    log_qz = Normal(mu, sigma).log_prob(mu).sum()
    log_pz = Normal(0, 1).log_prob(mu).sum()

    kl_div = log_qz - log_pz

    elbo = log_px - kl_div

    loss = -elbo.mean()

    return {"MSE": log_px, "KLD": kl_div, "loss": loss}


