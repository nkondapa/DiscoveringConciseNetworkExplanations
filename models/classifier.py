import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl
from models.utils import construct_model, modify_model_output_layer, construct_old_model


class ClassificationModel(pl.LightningModule):
    def __init__(self, model_type, num_classes, model_kwargs=None, optim_kwargs=None):
        super(ClassificationModel, self).__init__()
        self.save_hyperparameters()
        # Instantiate the model
        model = construct_model(model_type, **model_kwargs)
        model = modify_model_output_layer(model, num_classes)
        # model = construct_old_model(model_type, **model_kwargs)
        self.model = model
        self.model_kwargs = model_kwargs
        self.optim_kwargs = optim_kwargs

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        lr = self.optim_kwargs['lr']
        num_epochs = self.optim_kwargs['num_epochs']
        # optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = nn.functional.cross_entropy(outputs, targets)
        self.log('train_loss', loss, prog_bar=True)
        self.log('lr', self.optimizers().param_groups[0]['lr'], prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = nn.functional.cross_entropy(outputs, targets)
        acc = (outputs.argmax(dim=1) == targets).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
