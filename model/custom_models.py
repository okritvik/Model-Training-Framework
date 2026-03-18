from typing import Any
import pytorch_lightning as pl
from torch import optim, nn
import torch.nn.functional as torch_functions
import torch

class CustomModel(pl.LightningModule):
    """Class that provides template for the user to create a custom deep learning model

    Args:
        pl (pytorch_lightning): Pytorch Lightning module
    """
    def __init__(self, input_size, log_epoch = True, log_step = False, gpu_enabled = False):
        """Constructor to create an object of the class CustomModel

        Args:
            input_size (tuple): [(batch size, data shape)]
            log_epoch (bool, optional): [Enables log for each epoch]. Defaults to True.
            log_step (bool, optional): [Enables log for each step of training]. Defaults to False.
            gpu_enabled (bool, optional): [Flag to enable or disable GPU use for model training]. Defaults to False.
        """
        # Initialize class level variables - USER DEFINED
        super(CustomModel, self).__init__()
        self.input_size = input_size
        self.log_epoch = log_epoch
        self.log_step = log_step
        self.gpu_enabled = gpu_enabled
        
        # Model Architecture
        self.conv1 = nn.Conv2d(1, 128, 5) # Output [128, input_size]
        self.relu1 = nn.ReLU() # Output [32, input_size]
        self.conv2 = nn.Conv2d(128, 32, 3) # Output [32, input_size]
        self.dense1 = nn.Flatten()
        self.fc1 = nn.Linear(15488, 10) # output 10
        
        # lists to store the validation step and test step 
        self.validation_step_prediction = []
        self.validation_step_truth = []
        self.test_step_prediction = []
        self.test_step_truth = []
    
    def custom_loss_function(self, predicted_outputs, labels):
        """User defined loss function

        Args:
            predicted_outputs (_type_): output from the model
            labels (_type_): ground truth / labels

        Returns:
            _type_: total loss
        """
        return torch_functions.cross_entropy(predicted_outputs, labels)
    
    def forward(self, x):
        """Forward step

        Args:
            x (_type_): data or images

        Returns:
            _type_: output of the network after forward step
        """
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.dense1(x)
        x = self.fc1(x)
        return x
    
    def training_step(self, batch, batch_index):
        """Training step

        Args:
            batch (_type_): batch of data (example: 4,128,128,3 - 4 images of size 128x128x3)
            batch_index (_type_): index of the batch from the training data.

        Returns:
            _type_: loss
        """
        X, Y = batch # X is input images, Y is Labels
        output = self.forward(X) # Forward Pass
        loss = self.custom_loss_function(output, Y) # Calculate the Loss
        
        # Log the loss into TensorBoard
        self.log('train_loss', loss, prog_bar = True, on_epoch = self.log_epoch, on_step = self.log_step, logger = True)
        return loss
    
    def validation_step(self, batch, batch_index):
        """Validation step

        Args:
            batch (_type_): batch of data (example: 4,128,128,3 - 4 images of size 128x128x3)
            batch_index (_type_): index of the batch from the validation data. (Split from the train dataset)

        Returns:
            _type_: Validation loss
        """
        X, Y = batch # X is input images, Y is Labels
        outputs = self.forward(X) # Forward Pass
        loss = self.custom_loss_function(outputs, Y) # Calculate the Loss
        
        # Log the loss into TensorBoard
        self.log('validation_loss', loss, prog_bar = True, logger = True, on_epoch=self.log_epoch, on_step=self.log_step)
        return loss
    
    def test_step(self, batch, batch_index):
        """Testing step

        Args:
            batch (_type_): batch of data (example: 4,128,128,3 - 4 images of size 128x128x3)
            batch_index (_type_): index of the batch from the test data.

        Returns:
            _type_: test loss
        """
        X, Y = batch # X is input images, Y is Labels
        outputs = self.forward(X) # Forward Pass
        loss = self.custom_loss_function(outputs, Y) # Calculate the Loss
        
        # Log the loss into TensorBoard
        self.log('test_loss_step', loss, prog_bar = True, logger = True, on_epoch=self.log_epoch, on_step=self.log_step)
        return loss
    
    def configure_optimizers(self):
        """Configure user defined optimizer

        Returns:
            _type_: Optimizer
        """
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
    
        
    