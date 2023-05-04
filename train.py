import torch
import lightning.pytorch as pl
from segmentation import nll_loss, avg_sample_acc, SegmentationDataLoader
from models import CustomWav2Vec2Segmentation


class PTLModel(pl.LightningModule):
    
    def __init__(self,model):
        super().__init__()
        self.model = model
        self.save_hyperparameters()
    
    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        logits = outputs.get("logits")
        labels = batch.get("labels")
        loss = nll_loss(logits,labels)
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        logits = outputs.get("logits")
        labels = batch.get("labels")
        loss = nll_loss(logits,labels)
        self.log("val_loss", loss)
    
    def test_step(self, batch, batch_idx):
        scores = {}
        outputs = self.model(**batch)
        predictions = np.argmax(outputs.get("logits"),axis=-1) 
        reference = inputs.get("labels")
        scores.update(avg_sample_acc(predictions,reference))
        for k,v in scores.items():
            self.log(f"test_{k}", v)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def train(config):
    
    process_config(config,)
    
    dataloaders = {}
    for set_ in ['train','valid']:
        dataloader_config['dataset_config']['set'] = set_
        dataloaders[set_] = SegmentationDataLoader(**dataloader_config)
     
    train_loader = SegmentationDataLoader(_set='train',**config)
    valid_loader = SegmentationDataLoader(_set='dev',**config)
    
    model = CustomWav2Vec2Segmentation(**config)
    ptlmodel = PTLModel(model)
    
    trainer = pl.Trainer(default_root_dir=os.path.join(output_dir,'logs') )
    trainer.fit(model, train_loader, valid_loader)
    return 


def test(config):

    dataloader_config['dataset_config']['set'] = 'test'
    test_loaders = SegmentationDataLoader(**dataloader_config)
    
    ptlmodel = PTLModel.load_from_checkpoint(os.path.join(output_dir,'logs',"checkpoint.ckpt"))
    model.eval()
    
    return 
    


    
