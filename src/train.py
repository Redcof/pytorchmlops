from dotenv import load_dotenv
from torchvision.models import VGG11_Weights

load_dotenv()  # make sure python working directory to project root

import lightning as L
import mlflow.pytorch
import torch
import torchvision
from mlflow import MlflowClient
from torch.nn import functional as F
from torchmetrics import Accuracy, F1Score

from dataset import MyDataModule

IMG_SIZE = 224
NUM_LABELS = 6


class CNNModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        model = torchvision.models.vgg11(weights=VGG11_Weights.IMAGENET1K_V1)
        model.classifier[-1] = torch.nn.Linear(4096, NUM_LABELS)
        self.model = model
        self.train_accuracy = Accuracy("multilabel", num_labels=NUM_LABELS)
        self.train_f1score = F1Score("multilabel", num_labels=NUM_LABELS)
        self.val_accuracy = Accuracy("multilabel", num_labels=NUM_LABELS)
        self.val_f1score = F1Score("multilabel", num_labels=NUM_LABELS)

    def forward(self, x):
        return torch.sigmoid(self.model(x))

    def training_step(self, batch, batch_nb):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        # pred = logits.argmax(dim=1)
        acc = self.train_accuracy(logits, y)
        f1_score = self.train_f1score(logits, y)

        # PyTorch `self.log` will be automatically captured by MLflow.
        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", acc, on_epoch=True)
        self.log("train_f1_score", f1_score, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        # pred = logits.argmax(dim=1)
        acc = self.val_accuracy(logits, y)
        f1_score = self.val_f1score(logits, y)

        # PyTorch `self.log` will be automatically captured by MLflow.
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_acc", acc, on_epoch=True)
        self.log("val_f1_score", f1_score, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.02)


def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print(f"run_id: {r.info.run_id}")
    print(f"artifacts: {artifacts}")
    print(f"params: {r.data.params}")
    print(f"metrics: {r.data.metrics}")
    print(f"tags: {tags}")


if __name__ == '__main__':
    # Initialize our model.
    cnn_model = CNNModel()

    # Load dataset.
    datamodule = MyDataModule(im_size=(IMG_SIZE, IMG_SIZE), ratios=(0.65, 0.2, 0.15))
    # Initialize a trainer.
    trainer = L.Trainer(max_epochs=5, accelerator='gpu', num_sanity_val_steps=1)

    # mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    mlflow.set_experiment("pytorch_mlflow")
    # Auto log all MLflow entities
    mlflow.pytorch.autolog()

    # Train the model.
    with mlflow.start_run() as run:
        trainer.fit(cnn_model, datamodule)

    # Fetch the auto logged parameters and metrics.
    print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
