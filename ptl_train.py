import torch
import lightning.pytorch as pl
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from .segmentation import nll_loss, avg_sample_acc


class PTLModel(pl.LightningModule):

    def __init__(self,model):
        super().__init__()
        self.model = model

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
        scores = self._compute_metrics(logits,labels)
        for k,v in scores.items():
            self.log(f"val_{k}", v)

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

    def _compute_metrics(self,logits,labels):
        scores = {}
        acc = avg_sample_acc(pred,labels)
        scores.update(acc)
        return scores 


def raytune_objective(config):

    # early stopping and mornitoring 
    # hp

    train_loader = SegmentationDataLoader(_set='train',**config)
    valid_loader = SegmentationDataLoader(_set='dev',**config)

    model = CustomWav2Vec2Segmentation(**config)
    ptlmodel = PTLModel(model)

    metrics = {"loss": "ptl/val_loss", "acc": "ptl/val_accuracy"}

    trainer = pl.Trainer(
        max_epochs=config['max_epochs'],
        gpus=config['num_gpus'],
        progress_bar_refresh_rate=0,
        callbacks=[TuneReportCallback(metrics, on="validation_end")]
    )

    trainer.fit(ptlmodel, train_dataloaders=train_loader, val_dataloaders=valid_loader)


def raytune_job(config)
    raytune_config = {}
    config.update(raytune_config)

    objective_metric = config['objective_metric']
    num_samples = config['num_samples']
    cpu_per_trial = config.get('cpu_per_trial',1)
    gpu_per_trial = config.get('gpu_per_trial',1)

    analysis = tune.run(
        objective,
        resources_per_trial={
            "cpu": cpu_per_trial,
            "gpu": gpus_per_trial
        },
        metric=objective_metric,
        mode="max",
        config=config,
        num_samples=num_samples,
        name="tune_job"
    )

    print(analysis.best_config)