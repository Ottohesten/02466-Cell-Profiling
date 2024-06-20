from torch.utils.data import DataLoader, Dataset # importing packages
from torchvision import datasets
import torch
import numpy as np
from sklearn.model_selection import train_test_split


class MNIST(Dataset):
    def __init__(self, root="processed_data/", train=True, transform=None, download=True):
        self.dataset = datasets.MNIST(root, train=train, transform=transform, download=download)
        self.transform = transform


    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, label
    

def get_dataloader(train_batch_size, test_batch_size, transform=None):
    train_loader = DataLoader(MNIST(train=True, transform=transform), batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(MNIST(train=False, transform=transform), batch_size=test_batch_size, shuffle=False)
    return train_loader, test_loader


def npy_loader(path):
    """
    load a npy file and change the dtype to int8
    """
    sample = np.load(path)

    sample = (sample / np.max(sample) * 255).astype(np.uint8)

    sample = torch.from_numpy(sample)
    # make channel the first dimension
    sample = sample.permute(2, 0, 1)

    return sample




class OwnDataset(Dataset):
    def __init__(self, transform=None, train=True, path="data_subset"):
        self.transform = transform

        if transform is not None:
            self.dataset = datasets.DatasetFolder(path, loader=npy_loader, extensions=('.npy',), transform=transform)
        else:
            self.dataset = datasets.DatasetFolder(path, loader=npy_loader, extensions=('.npy',))
    
        # if train:
        #     self.dataset.samples = self.dataset.samples[:int(0.8*len(self.dataset))]
        # else:
            # self.dataset.samples = self.dataset.samples[int(0.8*len(self.dataset)):]

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx] # TODO: label seems to be coming out of thin air. the correct label is in metadata file | label comes from datasetFolder find_classes method, i think this is the wrong way to get the dataset
        return img, label
    

def make_train_test_val_split(dataset: OwnDataset):
    train_idx, test_idx = train_test_split(range(len(dataset)), test_size=0.2)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.2)
    # train_idx, test_idx = train_test_split(range(len(dataset)), test_size=0.2, stratify=dataset.dataset.targets)
    # train_idx, val_idx = train_test_split(train_idx, test_size=0.2, stratify=[dataset.dataset.targets[i] for i in train_idx])
    
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    
    return train_dataset, test_dataset, val_dataset



def make_small_subset(dataset: OwnDataset, n=1000):
    idx = np.random.choice(range(len(dataset)), n, replace=False)
    subset = torch.utils.data.Subset(dataset, idx)
    return subset


if __name__ == "__main__":
    print("ran dataset_tools.py as main file")

    data_train = OwnDataset()

    train, test, val = make_train_test_val_split(data_train)


    




