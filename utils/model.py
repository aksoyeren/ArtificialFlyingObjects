#from progress import LitProgressBar
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

class Model(pl.LightningModule): 
    def __init__(self, model, optimizer=None, criterion=None, train_metrics=None, validation_metrics=None,**hparams):
        super().__init__() 
        self.model = model
        self.optimizer = optimizer["type"](model.parameters(),**optimizer["args"])
        self.criterion = criterion
        
        # These have internal memory
        #self.train_metric = pl.metrics.Accuracy(compute_on_step=False)
        #self.val_metric = pl.metrics.Accuracy(compute_on_step=False)
        self.train_metrics = train_metrics
        self.validation_metrics = validation_metrics
    def forward(self, x):
        return self.model(x)
        
    def training_step(self, batch: dict, batch_idx: int) -> dict:
        x, target = batch
        logits = self.forward(x)
        loss = self.criterion(logits,target)
        preds = F.softmax(logits,dim=1)
        self.log('Train_loss',loss)
        self.train_metrics(preds, target)
        return {'loss':loss}
    
    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        x, target = batch
        logits = self.forward(x)

        loss = self.criterion(logits,target)
        preds = F.softmax(logits,dim=1)
 
        self.validation_metrics(preds, target)
        self.log('Validation_loss',loss)
        return {'val_loss':loss}
    
    def training_epoch_end(self, outputs):
        self.log_dict(self.train_metrics.compute())
        
    def validation_epoch_end(self, outputs):
        self.log_dict(self.validation_metrics.compute())
    
    def test_step(self, batch: dict, batch_idx: int) -> dict:
        x, target = batch
        logits = self.forward(x)

        loss = self.criterion(logits,target)
        preds = F.softmax(logits,dim=1)
 
        return {'test_loss':loss}
        
    def configure_optimizers(self):
        # Note: dont use list if only one item.. Causes silent crashes
        #optimizer = torch.optim.Adam(self.model.parameters())
        return self.optimizer
