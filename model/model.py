from typing import Any
import lightning as pl
from torch import optim, nn, utils, Tensor
import torch.nn.functional as torch_functions

class Model(pl.LightningModule):
    def __init__(self, input_size, log_epoch, log_step):
        super().__init__()
        self.input_size = input_size
        self.log_epoch = log_epoch
        self.log_step = log_step
        self.conv1 = nn.Conv2d(1, 128, 5) # Output [128, input_size]
        self.relu1 = nn.ReLU() # Output [32, input_size]
        self.conv2 = nn.Conv2d(128, 32, 3) # Output [32, input_size]
        self.fc1 = nn.Linear(32*input_size, 10) # output 10
    
    def custom_loss_function(self, predicted_outputs, labels):
        torch_functions.cross_entropy(predicted_outputs, labels)
    
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
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
    
        
    