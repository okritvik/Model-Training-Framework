from typing import Any
import pytorch_lightning as pl
from torch import optim, nn
import torch.nn.functional as torch_functions
import torch

class CustomModel(pl.LightningModule):
    def __init__(self, input_size, log_epoch = True, log_step = False):
        # Initialize class level variables
        super(CustomModel,self).__init__()
        self.input_size = input_size
        self.log_epoch = log_epoch
        self.log_step = log_step
        
        # Model Architecture
        self.conv1 = nn.Conv2d(1, 128, 5) # Output [128, input_size]
        self.relu1 = nn.ReLU() # Output [32, input_size]
        self.conv2 = nn.Conv2d(128, 32, 3) # Output [32, input_size]
        self.fc1 = nn.Linear(32, 10) # output 10
    
    def custom_loss_function(self, predicted_outputs, labels):
        return torch_functions.softmax(predicted_outputs, labels)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.fc1(x)
        return x
    
    def training_step(self, batch, batch_index):
        X, Y = batch # X is input images, Y is Labels
        output = self.forward(X) # Forward Pass
        loss = self.custom_loss_function(output, Y) # Calculate the Loss
        
        # Log the loss into TensorBoard
        self.log('train_loss', loss, prog_bar = True, on_epoch = self.log_epoch, on_step = self.log_step, logger = True)
        return loss
    
    def validation_step(self, batch, batch_index):
        X, Y = batch # X is input images, Y is Labels
        output = self.forward(X) # Forward Pass
        loss = self.custom_loss_function(output, Y) # Calculate the Loss
        
        # Log the loss into TensorBoard
        self.log('validation_loss', loss, prog_bar = True, logger = True)
        return loss
    
    def test_step(self, batch, batch_index):
        X, Y = batch # X is input images, Y is Labels
        output = self.forward(X) # Forward Pass
        loss = self.custom_loss_function(output, Y) # Calculate the Loss
        
        # Log the loss into TensorBoard
        self.log('test_loss_step', loss, prog_bar = True, logger = True)
        return loss
    
    def on_validation_epoch_end(self, outputs):
        avg_validation_loss = torch.stack([x for x in outputs]).mean()
        self.log('average_validation_loss', avg_validation_loss, prog_bar=True, logger=True)
        
    def on_test_epoch_end(self, outputs):
        avg_test_loss = torch.stack([x for x in outputs]).mean()
        self.log('average_test_loss', avg_test_loss, prog_bar=True, logger=True)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
    
        
    