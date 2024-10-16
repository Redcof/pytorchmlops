import random

import numpy as np
from dotenv import load_dotenv
from torchvision.models import EfficientNet_B0_Weights

load_dotenv()  # make sure python working directory to project root

import lightning
import mlflow.pytorch
import torch
from torch.nn.functional import cross_entropy
import torchvision
from mlflow import MlflowClient
from torchmetrics import Accuracy, F1Score

from dataset import MyDataModule
import logging

logger = logging.getLogger(__file__)
# create a formatter object
Log_Format = "%(asctime)s %(name)s [%(levelname)s]: %(message)s"
formatter = logging.Formatter(fmt=Log_Format)

# Add custom handler with format to this logger
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)


class CNN_pl_module(lightning.LightningModule):
    def __init__(self, num_labels, criterion, optimizer, lr):
        super().__init__()
        # model
        model = torchvision.models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_labels)
        self.model = model
        # others
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr = lr
        # metrics
        self.train_accuracy = Accuracy("multilabel", num_labels=num_labels)
        self.train_f1score = F1Score("multilabel", num_labels=num_labels)
        self.val_accuracy = Accuracy("multilabel", num_labels=num_labels)
        self.val_f1score = F1Score("multilabel", num_labels=num_labels)
        self.test_accuracy = Accuracy("multilabel", num_labels=num_labels)
        self.test_f1score = F1Score("multilabel", num_labels=num_labels)

    def forward(self, x):
        return torch.sigmoid(self.model(x))

    def training_step(self, batch, batch_nb):
        x, y = batch
        probas = self(x)
        loss = self.criterion(probas, y)
        acc = self.train_accuracy(probas, y)
        f1_score = self.train_f1score(probas, y)

        # PyTorch `self.log` will be automatically captured by MLflow.
        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", acc, on_epoch=True)
        self.log("train_f1_score", f1_score, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        probas = self(x)
        loss = self.criterion(probas, y)
        acc = self.val_accuracy(probas, y)
        f1_score = self.val_f1score(probas, y)

        # PyTorch `self.log` will be automatically captured by MLflow.
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_acc", acc, on_epoch=True)
        self.log("val_f1_score", f1_score, on_epoch=True)
        return loss

    def test_step(self, batch, batch_nb):
        x, y = batch
        probas = self(x)
        loss = self.criterion(probas, y)
        acc = self.val_accuracy(probas, y)
        f1_score = self.val_f1score(probas, y)

        # PyTorch `self.log` will be automatically captured by MLflow.
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", acc, on_epoch=True)
        self.log("test_f1_score", f1_score, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return self.optimizer(self.model.parameters(), lr=self.lr)


def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    logger.info(f"run_id: {r.info.run_id}")
    logger.info(f"artifacts: {artifacts}")
    logger.info(f"params: {r.data.params}")
    logger.info(f"metrics: {r.data.metrics}")
    logger.info(f"tags: {tags}")


if __name__ == '__main__':
    # setlogger
    log_level = logging.DEBUG
    logger.setLevel(log_level)
    from lightning import _logger as lightning_console_logger

    lightning_console_logger.setLevel(log_level)
    # define hyperparameters
    IMG_SIZE = 224
    NUM_LABELS = 6
    BATCH_SIZE = 16
    RANDOM_SEED = 37
    CRITERION = cross_entropy
    OPTIMIZER = torch.optim.Adam
    LR = 0.02

    # set random seed
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Initialize our model.
    cnn_model = CNN_pl_module(NUM_LABELS, CRITERION, OPTIMIZER, LR)

    # Load dataset.
    datamodule = MyDataModule(im_size=(IMG_SIZE, IMG_SIZE), ratios=(0.65, 0.2, 0.15), batch_size=BATCH_SIZE)
    # Initialize a trainer.
    # checkpoint_callback = ModelCheckpoint(save_top_k=0)
    trainer = lightning.Trainer(max_epochs=5, accelerator='gpu',
                                callbacks=[],
                                num_sanity_val_steps=1)

    mlflow.set_experiment("pytorch_mlflow")
    mlflow.pytorch.autolog(log_models=False, silent=False,
                           checkpoint=False, checkpoint_monitor="val_f1_score")

    # Train the model.
    with mlflow.start_run() as run:
        trainer.fit(cnn_model, datamodule)
        trainer.test(cnn_model, datamodule)

    # Fetch the auto logged parameters and metrics.
    print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
