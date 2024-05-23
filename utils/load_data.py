import os
from torchvision.datasets import MNIST, ImageNet, CIFAR10, Cityscapes, CocoDetection, Kitti
from torchvision.transforms import ToTensor
from torch import utils, Tensor
from torch.utils.data import DataLoader, random_split

def load_mnist(to_device = False, validation_split=0.1, batch_size=1):
    if to_device:
        train_dataset = MNIST(os.getcwd()+"/../dataset/available_datasets", train=True, download=True, transform=ToTensor())
        test_dataset = MNIST(os.getcwd()+"/../dataset/available_datasets", train=False, download=True, transform=ToTensor())
    else:
        train_dataset = MNIST(os.getcwd()+"/../dataset/available_datasets", train=True, download=True)
        test_dataset = MNIST(os.getcwd()+"/../dataset/available_datasets", train=False, download=True)
    
    # Split the data into training and validation.
    train_samples = len(train_dataset)
    validation_samples = int(train_samples * validation_split)
    train_samples -= validation_samples
    
    train_data, validation_data = random_split(train_dataset, [train_samples, validation_samples]) 
    
    train_loader = DataLoader(train_data, batch_size=batch_size)
    validation_loader = DataLoader(validation_data, batch_size=batch_size)
    test_loader = DataLoader(test_dataset)
    img, _ = train_dataset[0]
    
    return train_loader, validation_loader, test_loader, img.size()
    