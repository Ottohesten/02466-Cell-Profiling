import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from inference import Inference
from sklearn.preprocessing import StandardScaler
import numpy as np
import plotly.express as px
import pandas as pd

FIGURE_DIR = "figures/"


def show_img(img):
    plt.imshow(img.permute(1, 2, 0))
    plt.show()


def loss_plots(plot_data: dict, title: str = None, invidual_plots: bool = False):
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
        if title:
            plt.title(title)
        plt.show()

def accuracy_plots(plot_data: dict, invidual_plots: bool = False):
    """
    Plots the accuracy from the dictionary
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



def plot_latent_train(output, reduction_method="tsne"):
    mu = output["mu"].detach().cpu().numpy()
    sigma = output["sigma"].detach().cpu().numpy()
    z = output["z"].detach().cpu().numpy()

    scale_factor = 2.0

    if z.shape[1] > 2:
        if reduction_method == "tsne":
            # use t-sne to reduce of the latent space to 2 dimensions
            z = TSNE(n_components=2).fit_transform(z)
            mu = TSNE(n_components=2).fit_transform(mu)
        elif reduction_method == "umap":
            import umap
            z = umap.UMAP(n_components=2).fit_transform(z)
            mu = umap.UMAP(n_components=2).fit_transform(mu)
        elif reduction_method == "pca":
            from sklearn.decomposition import PCA
            z = PCA(n_components=2).fit_transform(z)
            mu = PCA(n_components=2).fit_transform(mu)
        else:
            raise ValueError("reduction_method must be one of 'tsne', 'umap', 'pca'")

    
    
    plt.figure(figsize=(8, 6))
    plt.scatter(mu[:, 0], mu[:, 1], c="r", label="mu")



    # plot of std deviations of the latent variables
    ellipses = [plt.matplotlib.patches.Ellipse((mu[i, 0], mu[i, 1]), sigma[i, 0] * scale_factor, sigma[i, 1] * scale_factor, alpha=0.3, fill=False) for i in range(z.shape[0])]

    for e in ellipses:
        plt.gca().add_artist(e)


    plt.legend()
    plt.show()


def plot_latent(inference: Inference, method="tsne", keys=[str(i) for i in range(13)]):
    samples = inference.samples
    scaler = StandardScaler()

    for key, val in samples.items():
        if key not in keys:
            continue
        mu = val["mu"]
        mu_scaled = scaler.fit_transform(mu)

        if method == "tsne":
            tsne = TSNE(n_components=2)
            z = tsne.fit_transform(mu_scaled)
        else:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            z = pca.fit_transform(mu_scaled)
        
        plt.scatter(z[:,0], z[:,1], label=key)
    plt.legend()
    plt.show()


def plot_split_stratification(dataset, train_subset, test_subset, val_subset):
    """
    Plot the stratification of the dataset
    """
    # check if the split is stratified
    # make the 4 plots side by side
    plt.figure(figsize=(24,6))
    plt.subplot(1,4,1)
    plt.title("Whole dataset")
    targets = np.array(dataset.dataset.targets)
    unique, counts = np.unique(targets, return_counts=True)
    plt.bar(unique, counts/len(dataset))
    plt.subplot(1,4,2)
    plt.title("Train dataset")
    train_targets = np.array(train_subset.dataset.dataset.targets)[train_subset.indices]
    train_unique, train_counts =  np.unique(train_targets, return_counts=True)
    plt.bar(train_unique, train_counts/len(train_subset))
    plt.subplot(1,4,3)
    plt.title("Test dataset")
    test_targets = np.array(test_subset.dataset.dataset.targets)[test_subset.indices]
    test_unique, test_counts =  np.unique(test_targets, return_counts=True)
    plt.bar(test_unique, test_counts/len(test_subset))
    plt.subplot(1,4,4)
    plt.title("Val dataset")
    val_targets = np.array(val_subset.dataset.dataset.targets)[val_subset.indices]
    val_unique, val_counts =  np.unique(val_targets, return_counts=True)
    plt.bar(val_unique, val_counts/len(val_subset))
    plt.show()

    # combine in the same plot using plotly express
    df = pd.DataFrame({
        "Dataset": ["Whole"]*len(unique) + ["Train"]*len(train_unique) + ["Test"]*len(test_unique) + ["Val"]*len(val_unique),
        "Class": np.concatenate([unique, train_unique, test_unique, val_unique]),
        "Percentage": np.concatenate([counts/len(dataset), train_counts/len(train_subset), test_counts/len(test_subset), val_counts/len(val_subset)])
    })

    fig = px.bar(df, x="Class", y="Percentage", color="Dataset", barmode="group")
    fig.show()
