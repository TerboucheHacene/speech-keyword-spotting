import torch
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
import os
import pytorch_lightning as pl
import torch.utils.data as tud


labels = [
    "marvin",
    "off",
    "left",
    "one",
    "cat",
    "on",
    "bed",
    "house",
    "sheila",
    "down",
    "happy",
    "visual",
    "five",
    "stop",
    "dog",
    "wow",
    "seven",
    "zero",
    "backward",
    "no",
    "eight",
    "three",
    "four",
    "tree",
    "nine",
    "go",
    "bird",
    "right",
    "yes",
    "up",
    "follow",
    "learn",
    "two",
    "forward",
    "six",
]


class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("..", download=False)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.join(self._path, line.strip()) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            # paths in load_list have "./" in them, need to add to walker too
            self._walker = [w for w in self._walker if "./" + w not in excludes]


def label_to_index(word, labels=labels):
    # Return the position of the word in labels
    return torch.tensor(labels.index(word))


def index_to_label(index, labels=labels):
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return labels[index]


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.0)
    return batch.permute(0, 2, 1)


def collate_fn(batch):
    # A data tuple has the form:
    # # # waveform, sample_rate, label, speaker_id, utterance_number
    tensors, targets = [], []
    # Gather in lists, and encode labels as indices
    for waveform, _, label, *_ in batch:
        tensors += [waveform]
        targets += [label_to_index(label)]
        # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)
    return tensors, targets


class SpeechDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=128, num_workers=0, transforms=None, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transorms = transforms

    def setup(self, stage=None) -> None:
        self.train_dataset = SubsetSC(subset="training")
        self.val_dataset = SubsetSC(subset="validation")
        self.test_dataset = SubsetSC(subset="testing")
        self.labels = sorted(list(set(datapoint[2] for datapoint in self.train_dataset)))

    def train_dataloader(self):
        train_loader = tud.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = tud.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return val_loader

    def test_dataloader(self):
        test_loader = tud.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_wrapper,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return test_loader
