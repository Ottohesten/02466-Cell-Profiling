import matplotlib.pyplot as plt
import torch


def imshow(img):
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
        if isinstance(output, tuple):
            x_hat = output[0]
        else:
            x_hat = output
            
    fig, ax = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25,5))
    for i in range(10):
        ax[0, i].imshow(images[i].cpu().detach().view(img_shape), cmap='gray')
        ax[1, i].imshow(x_hat[i].cpu().detach().view(img_shape), cmap='gray')
        ax[0, i].axis('off')
        ax[1, i].axis('off')



