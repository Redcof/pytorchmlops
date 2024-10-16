import logging

import lightning
import mlflow
import torch
import torchmetrics
import torchvision
from matplotlib import pyplot as plt
from torchmetrics import Accuracy, F1Score
from torchvision.models import EfficientNet_B0_Weights

from metrics import ClassificationReport

logger = logging.getLogger(__file__)


class CNN_pl_module(lightning.LightningModule):
    def __init__(self, num_labels, criterion, optimizer, lr, labels):
        super().__init__()
        # model
        model = torchvision.models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_labels)
        self.model = model
        # others
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr = lr
        self.labels = labels
        # metrics
        self.probability_threshold = 0.5
        self.train_accuracy = Accuracy("multilabel", num_labels=num_labels, threshold=self.probability_threshold)
        self.train_f1score = F1Score("multilabel", num_labels=num_labels, threshold=self.probability_threshold)
        self.val_accuracy = Accuracy("multilabel", num_labels=num_labels, threshold=self.probability_threshold)
        self.val_f1score = F1Score("multilabel", num_labels=num_labels, threshold=self.probability_threshold)
        self.test_accuracy = Accuracy("multilabel", num_labels=num_labels, threshold=self.probability_threshold)
        self.test_f1score = F1Score("multilabel", num_labels=num_labels, threshold=self.probability_threshold)
        self.train_classification_report = ClassificationReport(target_names=self.labels)
        self.val_classification_report = ClassificationReport(target_names=self.labels)
        # curves
        self.train_pr_curve = torchmetrics.PrecisionRecallCurve("multilabel", num_labels=num_labels)
        self.val_pr_curve = torchmetrics.PrecisionRecallCurve("multilabel", num_labels=num_labels)

    def forward(self, x):
        return torch.sigmoid(self.model(x))

    def training_step(self, batch, batch_nb):
        x, y = batch
        probas = self(x)
        loss = self.criterion(probas, y)
        acc = self.train_accuracy(probas, y)
        f1_score = self.train_f1score(probas, y)
        self.train_pr_curve.update(probas, y.to(torch.int8))
        self.train_classification_report.update(probas >= self.probability_threshold, y)

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
        self.val_pr_curve.update(probas, y.to(torch.int8))
        self.val_classification_report.update(probas >= self.probability_threshold, y.cpu())

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

    def on_fit_end(self):
        self._classification_report()
        self._plot_pr_curves()
        self._plot_f1_curves()

    def _classification_report(self):
        r = self.train_classification_report.compute()
        mlflow.log_dict(r, "classification_reports/train_classification_report.json")
        r = self.val_classification_report.compute()
        mlflow.log_dict(r, "classification_reports/val_classification_report.json")
        r = self.train_classification_report.compute(output_dict=False)
        logger.info(r)
        r = self.val_classification_report.compute(output_dict=False)
        logger.info(r)

    def _plot_pr_curves(self):
        plt.figure(figsize=(10, 7))
        # === === === === === === === === ===
        ax = plt.subplot(121)
        self.train_pr_curve.plot(ax=ax, score=True)
        ax.set_title('multilabel precision-recall curve [train]')
        # update legend texts with class name
        legend = plt.gca().get_legend()
        for text, new_label in zip(legend.get_texts(), self.labels):
            text.set_text(f"{new_label}:{text.get_text()}")
        # === === === === === === === === ===
        ax = plt.subplot(122)
        self.val_pr_curve.plot(ax=ax, score=True)
        ax.set_title('multilabel precision-recall curve [val]')
        # update legend texts with class name
        legend = plt.gca().get_legend()
        for text, new_label in zip(legend.get_texts(), self.labels):
            text.set_text(f"{new_label}:{text.get_text()}")
        # === === === === === === === === ===
        plt.savefig('precision-recall_curve.png', bbox_inches='tight')
        mlflow.log_artifact('precision-recall_curve.png', "graphs")
        plt.close()

    def _plot_f1_curves(self):
        plt.figure(figsize=(10, 7))
        # === === === === === === === === ===
        ax = plt.subplot(121)
        self.train_f1score.plot(ax=ax)
        ax.set_title('multilabel f1_score curve [train]')
        # update legend texts with class name
        # legend = plt.gca().get_legend()
        # for text, new_label in zip(legend.get_texts(), self.labels):
        #     text.set_text(f"{new_label}:{text.get_text()}")
        # === === === === === === === === ===
        ax = plt.subplot(122)
        self.val_f1score.plot(ax=ax)
        ax.set_title('multilabel fi_score curve [val]')
        # update legend texts with class name
        # legend = plt.gca().get_legend()
        # for text, new_label in zip(legend.get_texts(), self.labels):
        #     text.set_text(f"{new_label}:{text.get_text()}")
        # === === === === === === === === ===
        plt.savefig('f1_score_curve.png', bbox_inches='tight')
        mlflow.log_artifact('f1_score_curve.png', "graphs")
        plt.close()
