import pytorch_lightning as pl

import instant_nsr.models as models
from instant_nsr.systems.utils import parse_optimizer, parse_scheduler, update_module_step
from instant_nsr.utils.mixins import SaverMixin
from instant_nsr.utils.misc import config_to_primitive, get_rank


class BaseSystem(pl.LightningModule, SaverMixin):
    """
    Two ways to print to console:
    1. self.print: correctly handle progress bar
    2. rank_zero_info: use the logging module
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rank = get_rank()
        self.prepare()
        self.model = models.make(self.config.model.name, self.config.model)
    
    def prepare(self):
        pass

    def forward(self, batch):
        raise NotImplementedError
    
    def C(self, value):
        if isinstance(value, int) or isinstance(value, float):
            pass
        else:
            value = config_to_primitive(value)
            after_value = 0
            if not isinstance(value, list):
                raise TypeError('Scalar specification only supports list, got', type(value))
            if len(value) == 3:
                value = [0] + value
                start_step, start_value, end_value, end_step = value
            elif len(value) == 4:
                start_step, start_value, end_value, end_step = value
            elif len(value) == 5:
                # print("arrive")
                after_value = value[-1]
                start_step, start_value, end_value, end_step = value[0],value[1],value[2],value[3]
            
            elif len(value) == 6:
                current_step = self.global_step
                if current_step>value[3]:
                    start_step, start_value, end_value, end_step = value[3],value[2],value[4],value[5]
                else:
                    start_step, start_value, end_value, end_step = value[0],value[1],value[2],value[3]
            else:
                raise TypeError('Scalar specification only supports list with size 3-6, got', len(value))
            # assert len(value) == 4
            # if self.current_epoch > end_step and after_value > 0:
            #     value = after_value
            # else:
            if isinstance(end_step, int):
                current_step = self.global_step
                if current_step > end_step and after_value > 0:
                    value = after_value
                else:
                    value = start_value + (end_value - start_value) * max(min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0)
            elif isinstance(end_step, float):
                current_step = self.current_epoch
                if current_step > end_step and after_value > 0:
                    value = after_value
                else:
                    value = start_value + (end_value - start_value) * max(min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0)
            # import pdb;pdb.set_trace()
        return value
    
    def preprocess_data(self, batch, stage):
        pass

    """
    Implementing on_after_batch_transfer of DataModule does the same.
    But on_after_batch_transfer does not support DP.
    """
    def on_train_batch_start(self, batch, batch_idx, unused=0):
        self.dataset = self.trainer.datamodule.train_dataloader().dataset
        self.preprocess_data(batch, 'train')
        update_module_step(self.model, self.current_epoch, self.global_step)
    
    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
        self.dataset = self.trainer.datamodule.val_dataloader().dataset
        self.preprocess_data(batch, 'validation')
        update_module_step(self.model, self.current_epoch, self.global_step)
    
    def on_test_batch_start(self, batch, batch_idx, dataloader_idx):
        self.dataset = self.trainer.datamodule.test_dataloader().dataset
        self.preprocess_data(batch, 'test')
        update_module_step(self.model, self.current_epoch, self.global_step)

    def on_predict_batch_start(self, batch, batch_idx, dataloader_idx):
        self.dataset = self.trainer.datamodule.predict_dataloader().dataset
        self.preprocess_data(batch, 'predict')
        update_module_step(self.model, self.current_epoch, self.global_step)
    
    def training_step(self, batch, batch_idx):
        raise NotImplementedError
    
    """
    # aggregate outputs from different devices (DP)
    def training_step_end(self, out):
        pass
    """
    
    """
    # aggregate outputs from different iterations
    def training_epoch_end(self, out):
        pass
    """
    
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError
    
    """
    # aggregate outputs from different devices when using DP
    def validation_step_end(self, out):
        pass
    """
    
    def validation_epoch_end(self, out):
        """
        Gather metrics from all devices, compute mean.
        Purge repeated results using data index.
        """
        raise NotImplementedError

    def test_step(self, batch, batch_idx):        
        raise NotImplementedError
    
    def test_epoch_end(self, out):
        """
        Gather metrics from all devices, compute mean.
        Purge repeated results using data index.
        """
        raise NotImplementedError

    def export(self):
        raise NotImplementedError

    def configure_optimizers(self):
        optim = parse_optimizer(self.config.system.optimizer, self.model)
        ret = {
            'optimizer': optim,
        }
        if 'scheduler' in self.config.system:
            ret.update({
                'lr_scheduler': parse_scheduler(self.config.system.scheduler, optim),
            })
        return ret

