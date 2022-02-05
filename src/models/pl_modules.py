import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics


class ClassificationModel(pl.LightningModule):
    def __init__(self, model, transforms, learning_rate=0.001, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.transforms = transforms
        self.learning_rate = learning_rate

    def forward(self, x):
        if self.transforms is not None:
            x = self.transforms(x)
        y = self.model(x)
        return y

    def shared_step(self, batch, batch_index):
        data, targets = batch
        print(data.shape)
        output = self(data)
        loss = F.nll_loss(output.squeeze(), targets)
        acc = torchmetrics.functional.accuracy(output.squeeze(), targets)
        loss_dict = {"loss": loss, "accuracy": acc}
        return loss_dict

    def training_step(self, batch, batch_index):
        loss_dict = self.shared_step(batch, batch_index)
        for k, v in loss_dict.items():
            self.log(
                name="train_" + k,
                value=v,
                logger=True,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
        return loss_dict["loss"]

    def validation_step(self, batch, batch_index):
        loss_dict = self.shared_step(batch, batch_index)
        for k, v in loss_dict.items():
            self.log(
                name="val_" + k,
                value=v,
                logger=True,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        return loss_dict["loss"]

    def test_step(self, batch, batch_index):
        loss_dict = self.shared_step(batch, batch_index)
        for k, v in loss_dict.items():
            self.log(
                name="test_" + k,
                value=v,
                logger=True,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
        return loss_dict

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=0.0001
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        return [optimizer], [scheduler]
