from torch.utils.data import DataLoader, Dataset # importing packages
from torchvision import datasets
import torch
import numpy as np


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
    def __init__(self, transform=None, train=True):
        self.transform = transform

        if transform is not None:
            self.dataset = datasets.DatasetFolder("data_subset/singh_cp_pipeline_singlecell_images", loader=npy_loader, extensions=('.npy',), transform=transform)
        else:
            self.dataset = datasets.DatasetFolder("data_subset/singh_cp_pipeline_singlecell_images", loader=npy_loader, extensions=('.npy',))
    
        if train:
            self.dataset.samples = self.dataset.samples[:int(0.8*len(self.dataset))]
        else:
            self.dataset.samples = self.dataset.samples[int(0.8*len(self.dataset)):]

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, label