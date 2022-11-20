import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import (
    Accuracy,
    Precision,
    Recall,
    F1Score,
    ConfusionMatrix,
    AUROC,
    PrecisionRecallCurve,
    AveragePrecision,
    ROC,
)
from typing import Callable, Tuple, Dict, Any, Sequence
import wandb
import json
from keyword_detector.data.data_utils import LABELS


class SpeechLightningModel(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        transforms: Callable,
        num_classes: int,
        class_names: Sequence[str] = LABELS,
        learning_rate: float = 0.001,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.transforms = transforms
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.class_names = class_names

        # Initialize metrics
        self.train_metrics = nn.ModuleDict(
            {
                "accuracy": Accuracy(num_classes=self.num_classes, average="macro"),
                "precision": Precision(num_classes=self.num_classes, average="macro"),
                "recall": Recall(num_classes=self.num_classes, average="macro"),
                "f1": F1Score(num_classes=self.num_classes, average="macro"),
            }
        )
        self.val_metrics = nn.ModuleDict(
            {
                "accuracy": Accuracy(num_classes=self.num_classes, average="macro"),
                "precision": Precision(num_classes=self.num_classes, average="macro"),
                "recall": Recall(num_classes=self.num_classes, average="macro"),
                "f1": F1Score(num_classes=self.num_classes, average="macro"),
            }
        )
        self.test_metrics = nn.ModuleDict(
            {
                "accuracy": Accuracy(num_classes=self.num_classes, average="macro"),
                "precision": Precision(num_classes=self.num_classes, average="macro"),
                "recall": Recall(num_classes=self.num_classes, average="macro"),
                "f1": F1Score(num_classes=self.num_classes, average="macro"),
                "auroc": AUROC(num_classes=self.num_classes, average="macro"),
                "average_precision": AveragePrecision(
                    num_classes=self.num_classes, average="macro"
                ),
            }
        )
        self.test_metrics_non_scalar = nn.ModuleDict(
            {
                "confusion_matrix": ConfusionMatrix(num_classes=self.num_classes),
                "precision_recall_curve": PrecisionRecallCurve(
                    num_classes=self.num_classes
                ),
                "roc": ROC(num_classes=self.num_classes),
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.transforms is not None:
            x = self.transforms(x)
        y = self.model(x)
        return y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_index: Sequence[int]
    ) -> Dict[str, torch.Tensor]:
        data, targets = batch
        output = self(data)
        probs = F.softmax(output, dim=1)
        loss = F.cross_entropy(input=output, target=targets, reduction="mean")
        self.log(
            name="train/loss",
            value=loss,
            logger=True,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        # Update metrics
        for metric_name, metric in self.train_metrics.items():
            metric(probs, targets)
            self.log(
                name="train/" + metric_name,
                value=metric,
                logger=True,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
            )

        return {"loss": loss}

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_index: Sequence[int]
    ) -> Dict[str, torch.Tensor]:
        data, targets = batch
        output = self(data)
        probs = F.softmax(output, dim=1)
        loss = F.cross_entropy(input=output, target=targets, reduction="mean")
        self.log(
            name="val/loss",
            value=loss,
            logger=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        # Update metrics
        for metric_name, metric in self.val_metrics.items():
            metric(probs, targets)
            self.log(
                name="val/" + metric_name,
                value=metric,
                logger=True,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
        return {"loss": loss}

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_index: Sequence[int]
    ) -> Dict[str, torch.Tensor]:
        data, targets = batch
        output = self(data)
        probs = F.softmax(output, dim=1)
        loss = F.cross_entropy(input=output.squeeze(), target=targets, reduction="mean")
        # Update metrics
        for metric_name, metric in self.test_metrics.items():
            metric(probs, targets)
            self.log(
                name="test/" + metric_name,
                value=metric,
                logger=True,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
        # Update non-scalar metrics
        for metric_name, metric in self.test_metrics_non_scalar.items():
            metric(probs, targets)
        return {"loss": loss}

    def test_epoch_end(self, outputs: Dict[str, torch.Tensor]) -> None:

        probs = (
            torch.cat(
                getattr(self.test_metrics_non_scalar["precision_recall_curve"], "preds")
            )
            .cpu()
            .numpy()
        )
        y_true = (
            torch.cat(
                getattr(self.test_metrics_non_scalar["precision_recall_curve"], "target")
            )
            .cpu()
            .numpy()
        )
        self.logger.log_metrics(
            {
                "confusion_matrix": wandb.plot.confusion_matrix(
                    probs=probs,
                    y_true=y_true,
                    class_names=self.class_names,
                    title="Confusion Matrix",
                )
            }
        )

        self.logger.log_metrics(
            {
                "pr_curve": wandb.plot.pr_curve(
                    y_probas=probs,
                    y_true=y_true,
                    labels=self.class_names,
                    title="PR Curve",
                )
            }
        )
        self.logger.log_metrics(
            {
                "roc_curve": wandb.plot.roc_curve(
                    y_probas=probs,
                    y_true=y_true,
                    labels=self.class_names,
                    title="ROC Curve",
                )
            }
        )
        # save confusion matrix
        confusion_matrix = (
            self.test_metrics_non_scalar["confusion_matrix"]
            .compute()
            .cpu()
            .numpy()
            .tolist()
        )

        # save pr curve
        # first list is for precision and recall and threshold
        # second list contains class wise precision, recall and threshold
        precision_recall_curve = [
            [class_results.cpu().numpy().tolist() for class_results in item]
            for item in self.test_metrics_non_scalar["precision_recall_curve"].compute()
        ]

        roc_curve = [
            [class_results.cpu().numpy().tolist() for class_results in item]
            for item in self.test_metrics_non_scalar["roc"].compute()
        ]
        test_results = {
            "confusion_matrix": confusion_matrix,
            "precision_recall_curve": precision_recall_curve,
            "roc": roc_curve,
            "accuracy": self.test_metrics["accuracy"].compute().cpu().numpy().item(),
            "precision": self.test_metrics["precision"].compute().cpu().numpy().item(),
            "recall": self.test_metrics["recall"].compute().cpu().numpy().item(),
            "f1": self.test_metrics["f1"].compute().cpu().numpy().item(),
            "auroc": self.test_metrics["auroc"].compute().cpu().numpy().item(),
            "average_precision": self.test_metrics["average_precision"]
            .compute()
            .cpu()
            .numpy()
            .item(),
        }
        with open("artifacts/json/results.json", "w") as f:
            json.dump(test_results, f)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=0.0001
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
