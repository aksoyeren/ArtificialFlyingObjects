#from progress import LitProgressBar
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

class Model(pl.LightningModule): 
    """ """
    def __init__(self, model, optimizer=None, criterion=None, train_metrics=None, validation_metrics=None, test_metrics=None,**hparams):
        super().__init__() 
        self.model = model
        self.optimizer = optimizer["type"](model.parameters(),**optimizer["args"])
        self.criterion = criterion
        
        # These have internal memory
        #self.train_metric = pl.metrics.Accuracy(compute_on_step=False)
        #self.val_metric = pl.metrics.Accuracy(compute_on_step=False)
        self.train_metrics = train_metrics
        self.validation_metrics = validation_metrics
        self.test_metrics = test_metrics
        
    def forward(self, x):
        """

        :param x: 

        """
        return self.model(x)
        
    def training_step(self, batch: dict, batch_idx: int) -> dict:
        """

        :param batch: dict:
        :param batch_idx: int:

        """
        x, target = batch
        logits = self.forward(x)
        loss = self.criterion(logits,target)
        preds = F.softmax(logits,dim=1)
        self.log('Loss_Train',loss)
        if self.train_metrics != None: self.train_metrics(preds, target)
        return {'loss':loss}
    
    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        """

        :param batch: dict:
        :param batch_idx: int:

        """
        x, target = batch
        logits = self.forward(x)

        loss = self.criterion(logits,target)
        preds = F.softmax(logits,dim=1)
     
        if self.validation_metrics != None: self.validation_metrics(preds, target)
        self.log('loss_Validation',loss)
        return {'loss_Validation':loss}
    
    def test_step(self, batch: dict, batch_idx: int) -> dict:
        """

        :param batch: dict:
        :param batch_idx: int:

        """
        x, target = batch
        logits = self.forward(x)

        loss = self.criterion(logits,target)
        preds = F.softmax(logits,dim=1)
        
        if self.test_metrics != None: self.test_metrics(preds, target)
        self.log('loss_Test',loss)
        self.log_dict(self._is_metrics_float(self.test_metrics.compute()))
        return {'loss_Test':loss}

    def training_epoch_end(self, outputs):
        """

        :param outputs: 

        """
        if self.train_metrics != None: self.log_dict(self.train_metrics.compute())
        
    def validation_epoch_end(self, outputs):
        """

        :param outputs: 

        """
        if self.validation_metrics != None: self.log_dict(self.validation_metrics.compute())
    
    def _is_metrics_float(self, metrics:"dict") -> dict:
        """Check to ensure that the logger only logs single number variables.

        :param metrics: dict":

        """
        #print(metrics.items())
        return {key:value for key,value in metrics.items() if isinstance(value.dtype, float)}
            
       
        
    
    def configure_optimizers(self):
        """ """
        # Note: dont use list if only one item.. Causes silent crashes
        #optimizer = torch.optim.Adam(self.model.parameters())
        return self.optimizer
