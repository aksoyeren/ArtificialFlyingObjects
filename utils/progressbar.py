from pytorch_lightning.callbacks import progress
__all__ = ['LitProgressBar']

class LitProgressBar(progress.ProgressBarBase):
    """ """
#https://pytorch-lightning.readthedocs.io/en/latest/_modules/pytorch_lightning/callbacks/progress.html#ProgressBarBase.on_validation_batch_end
    def __init__(self):
        super().__init__()  # don't forget this :)
        self.enable()

    def disable(self):
        """Disable progressBar"""
        self._enable = False
    
    def enable(self):
        """Enable progressBar"""
        self._enable = True
        
    def on_epoch_start(self, trainer, pl_module):
        """

        :param trainer: param pl_module:
        :param pl_module: 

        """
        super().on_train_start(trainer, pl_module)

        print("",end="", flush=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """

        :param trainer: param pl_module:
        :param outputs: param batch:
        :param batch_idx: param dataloader_idx:
        :param pl_module: param batch:
        :param dataloader_idx: param batch:
        :param batch: 

        """
        super().on_train_batch_end(trainer, pl_module, outputs,batch, batch_idx) 
        
        con = f'Epoch {trainer.current_epoch+1} [{batch_idx+1:.00f}/{self.total_train_batches:.00f}] {self.get_progress_bar_dict(trainer)}'
        
        self._update(con)
        
    def _update(self,con:str) -> None:
        """Update console

        :param con: param con:str:
        :param con: str:
        :param con:str: 

        """

        print(con, end="\r", flush=True)
        
    def get_progress_bar_dict(self,trainer):
        """

        :param trainer: 

        """
        tqdm_dict = trainer.progress_bar_dict
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        return tqdm_dict