from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
import torch

"""The data needs to be arranged in a specific format.
In dataset directory, arrange your data as shown.
    - dataset
        - your_dataset
            - train (this folder contains training images/data)
            - val   (this folder contains validation images/data)
            - test  (this folder contains test images/data)
            - train_labels.txt
            - val_labels.txt
            - test_labels.txt
If the data is arranged in a different order, user needs to modify the setup function in
CustomDataloader class, and modify len, getitem functions in this class.

Returns:
    _type_: _description_
"""
class CustomDataset(Dataset):
    def __init__(self, data_path, labels_path, use_gpu=False) -> None:
        """Constructor for CustomDataset class

        Args:
            data_path (_type_): path of the train/val/test folder
            labels_path (_type_): path of the <train/test/val>_labels.txt
            use_gpu (_type_): boolean to convert the data to tensors if set to True. Defaults to False.
        """
        super().__init__()
        self.data_path = data_path
        self.labels_path = labels_path
        self.use_gpu = use_gpu

    def __len__(self) -> int:
        """Function to return the number of files present in the given directory

        Returns:
            int: number of images
        """
        self.images = os.listdir(self.data_path)
        self.labels = open(self.labels_path, "r").readlines()
        return len(self.images)
    
    def __getitem__(self, index):
        """Gets the data and label

        Args:
            index (int): index of the image

        Returns:
            tuple: image ndarray/tensor array, label tensor/integer
        """
        # Make sure to correct the datatype in the return statement according to the application. int/ndarray/list/tuple?
        # Currently the function is returning int.
        if self.use_gpu:
            return torch.tensor(plt.imread(self.data_path+"/"+self.images[index])), torch.tensor(int(self.labels[index]))
        else:
            return plt.imread(self.data_path+"/"+self.images[index]), int(self.labels[index])