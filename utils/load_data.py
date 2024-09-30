import os
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torchvision.datasets import MNIST, ImageNet, CIFAR10, Cityscapes, Kitti, CocoDetection
from torchvision.transforms import ToTensor
from torch import utils, Tensor
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from custom_dataset import CustomDataset

# TODO: For coco, the dataset needs to be downloaded, MS COCO API needs to be installed.
# TODO: Add num_workers as a parameter for the functions (defaults 1) to support multiprocessing
# TODO: Add class to load custom dataset (user needs to modify if required)

def load_mnist(to_device = False, validation_split=0.1, batch_size=1):
    """Download and return MNIST dataloaders.

    Args:
        to_device (bool, optional): Convert data for CUDA device. Defaults to False.
        validation_split (float, optional): Split for validation dataset. Defaults to 0.1.
        batch_size (int, optional): Batch size. Defaults to 1.

    Returns:
        _type_: train data loader, validation data loader, test data loader, size of the first batch 
    """
    my_transform = ToTensor()
    if to_device:
        train_dataset = MNIST(os.getcwd()+"/../dataset/available_datasets", train=True, download=True, transform=my_transform)
        test_dataset = MNIST(os.getcwd()+"/../dataset/available_datasets", train=False, download=True, transform=my_transform)
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
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    img, _ = train_dataset[0]
    
    return train_loader, validation_loader, test_loader, img.size()
    
def load_imagenet(to_device = False, validation_split=0.1, batch_size=1):
    """Load imagenet data and return IMAGENET dataloaders.

    Args:
        to_device (bool, optional): Convert data for CUDA device. Defaults to False.
        validation_split (float, optional): Split for validation dataset. Defaults to 0.1. (Not used)
        batch_size (int, optional): Batch size. Defaults to 1.

    Returns:
        _type_: train data loader, validation data loader
    """
    my_transform = ToTensor()
    if to_device:
        train_dataset = ImageNet(os.getcwd()+"/../dataset/available_datasets", split="train", transform=my_transform)
        validation_dataset = ImageNet(os.getcwd()+"/../dataset/available_datasets", split="val", transform=my_transform)
    else:
        train_dataset = ImageNet(os.getcwd()+"/../dataset/available_datasets", split="train")
        validation_dataset = ImageNet(os.getcwd()+"/../dataset/available_datasets", split="val")
    
    # Split the data into training and validation.
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size)

    return train_loader, validation_loader
    
def load_cifar10(to_device = False, validation_split=0.1, batch_size=1):
    """Download and return CIFAR10 dataloaders.

    Args:
        to_device (bool, optional): Convert data for CUDA device. Defaults to False.
        validation_split (float, optional): Split for validation dataset. Defaults to 0.1.
        batch_size (int, optional): Batch size. Defaults to 1.

    Returns:
        _type_: train data loader, validation data loader, test data loader, size of the first batch 
    """
    my_transform = ToTensor()
    if to_device:
        train_dataset = CIFAR10(os.getcwd()+"/../dataset/available_datasets", train=True, download=True, transform=my_transform)
        test_dataset = CIFAR10(os.getcwd()+"/../dataset/available_datasets", train=False, download=True, transform=my_transform)
    else:
        train_dataset = CIFAR10(os.getcwd()+"/../dataset/available_datasets", train=True, download=True)
        test_dataset = CIFAR10(os.getcwd()+"/../dataset/available_datasets", train=False, download=True)
    
    # Split the data into training and validation.
    train_samples = len(train_dataset)
    validation_samples = int(train_samples * validation_split)
    train_samples -= validation_samples
    
    train_data, validation_data = random_split(train_dataset, [train_samples, validation_samples]) 
    
    train_loader = DataLoader(train_data, batch_size=batch_size)
    validation_loader = DataLoader(validation_data, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    img, _ = train_dataset[0]
    
    return train_loader, validation_loader, test_loader, img.size()
    
def load_citiscapes(to_device = False, validation_split=0.1, batch_size=1, mode="fine"):
    """Load citiscapes data and return dataloaders. Citiscapes dataset must be downloaded from the official
    website.
    
    User needs to update the Citiscapes call with target_type if using. The current implementation
    of this api doesn't account the target_type for the dataset. 
    target_type (string or list, optional): Type of target to use, instance, semantic, polygon
        or color. Can also be a list to output a tuple with all specified target types. Defaults to instance

    Args:
        to_device (bool, optional): Convert data for CUDA device. Defaults to False.
        validation_split (float, optional): Split for validation dataset. Defaults to 0.1.
        batch_size (int, optional): Batch size. Defaults to 1.
        mode (str, optional): fine or coarse. Defaults to "fine".

    Returns:
        _type_: _description_
    """
    my_transform = ToTensor()
    if to_device:
        train_dataset = Cityscapes(os.getcwd()+"/../dataset/available_datasets", split="train", mode=mode, transform=my_transform)
        validation_dataset = Cityscapes(os.getcwd()+"/../dataset/available_datasets", split="val", mode=mode, transform=my_transform)
        if mode == "fine":
            test_dataset = Cityscapes(os.getcwd()+"/../dataset/available_datasets", split="test", transform=my_transform)
    else:
        train_dataset = Cityscapes(os.getcwd()+"/../dataset/available_datasets", split="train", mode=mode)
        validation_dataset = Cityscapes(os.getcwd()+"/../dataset/available_datasets", split="val", mode=mode)
        if mode == "fine":
            test_dataset = Cityscapes(os.getcwd()+"/../dataset/available_datasets", split="test")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size)
    img, _ = train_dataset[0]
    if mode == "fine":
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        return train_loader, validation_loader, test_loader, img.size()
    else:
        return train_loader, validation_loader, img.size()
    
def load_kitti(to_device = False, validation_split=0.1, batch_size=1):
    """Download and return CIFAR10 dataloaders.

    Args:
        to_device (bool, optional): Convert data for CUDA device. Defaults to False.
        validation_split (float, optional): Split for validation dataset. Defaults to 0.1.
        batch_size (int, optional): Batch size. Defaults to 1.

    Returns:
        _type_: train data loader, validation data loader, test data loader, size of the first batch 
    """
    my_transform = ToTensor()
    if to_device:
        train_dataset = Kitti(os.getcwd()+"/../dataset/available_datasets", train=True, download=True, transform=my_transform)
        test_dataset = Kitti(os.getcwd()+"/../dataset/available_datasets", train=False, download=True, transform=my_transform)
    else:
        train_dataset = Kitti(os.getcwd()+"/../dataset/available_datasets", train=True, download=True)
        test_dataset = Kitti(os.getcwd()+"/../dataset/available_datasets", train=False, download=True)
    
    # Split the data into training and validation.
    train_samples = len(train_dataset)
    validation_samples = int(train_samples * validation_split)
    train_samples -= validation_samples
    
    train_data, validation_data = random_split(train_dataset, [train_samples, validation_samples]) 
    
    train_loader = DataLoader(train_data, batch_size=batch_size)
    validation_loader = DataLoader(validation_data, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    img, _ = train_dataset[0]
    
    return train_loader, validation_loader, test_loader, img.size()

class CustomDataloader(pl.LightningDataModule):
    def __init__(self, data_path, num_workers, batch_size, to_device) -> None:
        super().__init__()
        self.data_path = data_path
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.to_device = to_device
    
    def prepare_data(self) -> None:
        # run on single core.
        # if the dataset needs to be downloaded, download in this function
        pass
    
    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = CustomDataset(data_path=self.data_path+"/train", labels_path=self.data_path+"/train_labels.txt", use_gpu=self.to_device)
        elif stage == "validate":
            self.val_dataset = CustomDataset(data_path=self.data_path+"/val", labels_path=self.data_path+"/val_labels.txt", use_gpu=self.to_device)
        elif stage == "test":
            self.test_dataset =CustomDataset(data_path=self.data_path+"/test", labels_path=self.data_path+"/test_labels.txt", use_gpu=self.to_device)
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    