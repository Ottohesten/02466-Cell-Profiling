# This is the main training file to be ran


def main():
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import transforms


    from dataset_tools import OwnDataset, make_train_test_val_split
    from collections import defaultdict

    cuda = torch.cuda.is_available()


    # load the dataset
    val = True

    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: x.view(-1)) # notice that we dont flatten when we are going to use CNN
    ])

    dataset = OwnDataset(transform=tf, path="/work3/s194101/labelled_data/")
    # dataset = OwnDataset(transform=tf)

    batch_size = 64
    train_subset, test_subset, val_subset = make_train_test_val_split(dataset)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, pin_memory=cuda, drop_last=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=True, pin_memory=cuda, drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=True, pin_memory=cuda, drop_last=True)


    # load the model
    from models import VAE_LAFARGE
    from loss_functions import loss_function
    model = VAE_LAFARGE(input_dim=(3,68,68), hidden_dim=512, latent_dim=256)

    if cuda:
        model.cuda()

    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # define dirs for the saving of model / data
    MODEL_NAME = f"{model.__class__.__name__}_latent{model.latent_dim}_"
    MODEL_DIR = "trained_models/"
    TRAIN_DATA_DIR = "train_data/"


    # training loop
    print("Starting training")
    num_epochs = 100

    train_loss = []
    train_mse_loss = []
    train_kld_loss = []
    val_loss = []
    val_mse_loss = []
    val_kld_loss = []
    best_loss = np.inf

    for epoch in range(num_epochs):
        batch_loss = []
        mse_batch_loss = []
        kld_batch_loss = []
        model.train()

        for x, y in train_loader:
            if cuda:
                x = x.cuda()

            optimizer.zero_grad()
            output = model(x)
            x_hat, mu, sigma = output["x_hat"], output["mu"], output["sigma"]
            loss_fn = loss_function(x, x_hat, mu, sigma)
            mse_loss = loss_fn["MSE"]
            kld_loss = loss_fn["KLD"]
            loss = loss_fn["loss"]

            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
            mse_batch_loss.append(mse_loss.item())
            kld_batch_loss.append(kld_loss.item())
            

        train_loss.append(np.mean(batch_loss))
        train_mse_loss.append(np.mean(mse_batch_loss))
        train_kld_loss.append(np.mean(kld_batch_loss))

        if val:
            model.eval()
            batch_loss = []
            batch_mse_loss = []
            batch_kld_loss = []
            for x, y in val_loader:
                if cuda:
                    x = x.cuda()

                output_val = model(x)
                x_hat, mu, sigma = output_val["x_hat"], output_val["mu"], output_val["sigma"]
                loss_fn = loss_function(x, x_hat, mu, sigma)
                loss = loss_fn["loss"]
                mse_loss = loss_fn["MSE"]
                kld_loss = loss_fn["KLD"]

                batch_loss.append(loss.item())
                batch_mse_loss.append(mse_loss.item())
                batch_kld_loss.append(kld_loss.item())

            val_loss.append(np.mean(batch_loss))
            val_mse_loss.append(np.mean(batch_mse_loss))
            val_kld_loss.append(np.mean(batch_kld_loss))

            if val_loss[-1] < best_loss:
                best_loss = val_loss[-1]
                torch.save(model.state_dict(), MODEL_DIR + MODEL_NAME + "hpc_best_model.pth")

        # print(f"Epoch {epoch+1}/{num_epochs}, loss: {train_loss[-1]}")
        print(f"Epoch {epoch+1}/{num_epochs}, loss: {train_loss[-1]}, mse_loss: {train_mse_loss[-1]}, kld_loss: {train_kld_loss[-1]}, val_loss: {val_loss[-1]}, val_mse_loss: {val_mse_loss[-1]}, val_kld_loss: {val_kld_loss[-1]}")

    # test the model
    # we evaluate model on test set
    test_loss = []
    test_mse_loss = []
    test_kld_loss = []
    model.eval()

    for x, y in test_loader:
        if cuda:
            x = x.cuda()

        output_test = model(x)
        x_hat, mu, sigma = output_test["x_hat"], output_test["mu"], output_test["sigma"]
        loss_fn = loss_function(x, x_hat, mu, sigma)
        mse_loss = loss_fn["MSE"]
        kld_loss = loss_fn["KLD"]
        loss = loss_fn["loss"]

        test_loss.append(loss.item())
        test_mse_loss.append(mse_loss.item())
        test_kld_loss.append(kld_loss.item())

    print(f"Test loss: {np.mean(test_loss)}, Test mse loss: {np.mean(test_mse_loss)}, Test kld loss: {np.mean(test_kld_loss)}")

    # save the data
    # make a dictionary with the losses as keys and the values as lists
    loss_data = {}
    loss_data["train_loss"] = train_loss
    loss_data["train_mse_loss"] = train_mse_loss
    loss_data["train_kld_loss"] = train_kld_loss
    loss_data["val_loss"] = val_loss
    loss_data["val_mse_loss"] = val_mse_loss
    loss_data["val_kld_loss"] = val_kld_loss
    loss_data["test_loss"] = np.mean(test_loss)
    loss_data["test_mse_loss"] = np.mean(test_mse_loss)
    loss_data["test_kld_loss"] = np.mean(test_kld_loss)

    df = pd.DataFrame(loss_data)
    try:
        old_df = pd.read_csv(TRAIN_DATA_DIR + MODEL_NAME + "loss_data.csv")
        df = pd.concat([old_df, df])
    except:
        pass
    df.to_csv(TRAIN_DATA_DIR + MODEL_NAME + "loss_data.csv", index=False)






if __name__ == '__main__':
    main()