import os
from model.custom_models import CustomModel
from utils import load_data
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl

# define batch size
batch = 4
# load the dataloaders with given batch size
train_loader, validation_loader, test_loader, size = load_data.load_mnist(to_device=True, batch_size=batch)

# input shape - append batch size
data_size_placeholder = tuple([batch] + list(size))

# Create a model instance
model = CustomModel(input_size=data_size_placeholder)

# Tensorboard logger for easy graphs of losses
tb_logger = TensorBoardLogger(save_dir=os.getcwd()+"/logs", name="mnist_train")

# save the chekpoints in the checkpoints folder
save_checkpoint = ModelCheckpoint(
    dirpath=os.getcwd()+"/checkpoints",  # Directory where the checkpoints will be saved
    filename='epoch-'+'{epoch}',  # Filename template
    save_top_k=1,  # Save the top 1 model
    monitor='validation_loss',  # Metric to monitor
    mode='min'  # Mode (minimize the monitored metric)
)


trainer = pl.Trainer(max_epochs=10, logger=tb_logger, callbacks=save_checkpoint)
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=validation_loader)

trainer.test(model,test_loader)

