import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics


class M5(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)


def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)


class ClassificationModel(pl.LightningModule):
    def __init__(self, model, transforms, learning_rate=0.001, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.transforms = transforms
        self.learning_rate = learning_rate
        self.accurcay = torchmetrics.Accuracy()

    def forward(self, input):
        pass

    def shared_step(self, batch):
        data, targets = batch
        data = self.transforms(data)
        output = self.model(data)
        loss = F.nll_loss(output.squeeze(), targets)
        accuracy = self.accurcay(output.squeeze(), targets)
        loss_dict = {"loss": loss, "accuracy": accuracy}
        return loss_dict

    def training_step(self, batch, batch_index):
        loss_dict = self.shared_step(batch)

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
        loss_dict = self.shared_step(batch)

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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=0.0001
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        return [optimizer], [scheduler]
