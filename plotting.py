import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch


def show_img(img):
    plt.imshow(img.permute(1, 2, 0))
    plt.show()


def loss_plots(plot_data: dict, invidual_plots: bool = False):
    """
    Plots the losses from the dictionary
    """
    if invidual_plots:
        for key, value in plot_data.items():
            plt.plot(value, label=key)
            plt.title(key)
            plt.grid()
            plt.show()

    else:
        for key, value in plot_data.items():
            plt.plot(value, label=key)
        plt.legend()
        plt.show()
    

def plot_random_images(model, n=10, cuda=False, img_shape=(28, 28)):
    """
    Generate n random latent space samples, and use the decoder to generate images
    """
    model.eval()
    with torch.no_grad():
        z = torch.randn(n, model.latent_dim)
        if cuda:
            z = z.cuda()
        images = model.decoder(z)

    if len(img_shape) > 2:
        fig, ax = plt.subplots(1, n, figsize=(25, 5))
        for i in range(n):
            ax[i].imshow(images[i].cpu().detach().view(img_shape).permute(1, 2, 0), cmap='gray')
            ax[i].axis('off')
    else:
        fig, ax = plt.subplots(1, n, figsize=(25, 5))
        for i in range(n):
            ax[i].imshow(images[i].cpu().detach().view(img_shape), cmap='gray')
            ax[i].axis('off')

    plt.show()

def plot_image_comparison(model, test_loader, cuda, img_shape=(28, 28)):
    """
    Plot the original image and the reconstructed image
    """
    model.eval()
    images, labels = next(iter(test_loader))

    with torch.no_grad():
        if cuda:
            images = images.cuda()
        output = model(images)
        x_hat = output["x_hat"]

    if len(img_shape) > 2:
        fig, ax = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25,5))
        for i in range(10):
            ax[0, i].imshow(images[i].cpu().detach().view(img_shape).permute(1, 2, 0), cmap='gray')
            ax[1, i].imshow(x_hat[i].cpu().detach().view(img_shape).permute(1, 2, 0), cmap='gray')
            ax[0, i].axis('off')
            ax[1, i].axis('off')
    else:
        fig, ax = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25,5))
        for i in range(10):
            ax[0, i].imshow(images[i].cpu().detach().view(img_shape), cmap='gray')
            ax[1, i].imshow(x_hat[i].cpu().detach().view(img_shape), cmap='gray')
            ax[0, i].axis('off')
            ax[1, i].axis('off')



def plot_latent(output):
    mu = output["mu"].detach().cpu().numpy()
    sigma = output["sigma"].detach().cpu().numpy()
    z = output["z"].detach().cpu().numpy()

    scale_factor = 2.0

    # if z.shape[1] > 2:
    #     z = TSNE(n_components=2).fit_transform(z)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(z[:, 0], z[:, 1], c="g", label="z")
    plt.scatter(mu[:, 0], mu[:, 1], c="r", label="mu")



    # plot of std deviations of the latent variables
    ellipses = [plt.matplotlib.patches.Ellipse((mu[i, 0], mu[i, 1]), sigma[i, 0] * scale_factor, sigma[i, 1] * scale_factor, alpha=0.3, fill=False) for i in range(z.shape[0])]

    for e in ellipses:
        plt.gca().add_artist(e)


    plt.legend()
    plt.show()

