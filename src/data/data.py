import torch
import torchaudio
import torch.utils.data as tud
import torch.nn.functional as F
from torchaudio.datasets import SPEECHCOMMANDS

import pytorch_lightning as pl

import os
import glob
import pandas as pd


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


class UrbanSoundDataset(tud.Dataset):
    """
    A rapper class for the UrbanSound8K dataset.
    """

    def __init__(
        self,
        annotation_file,
        audio_dir,
        folds,
        target_sample_rate=16000,
        num_samples=32000,
    ):
        self.audio_file = pd.read_csv(annotation_file)
        self.folds = folds
        self.audio_paths = glob.glob(audio_dir + "/*" + str(self.folds) + "/*")
        self.num_samples = num_samples
        self.target_sample_rate = target_sample_rate

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):

        audio_sample_path = self._get_audio_sample_path(idx)
        label = self._get_audio_sample_label(audio_sample_path)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        return signal, label

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, : self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_path(self, idx):
        audio_path = self.audio_paths[idx]
        return audio_path

    def _get_audio_sample_label(self, audio_path):
        audio_name = audio_path.split(sep="/")[-1]
        label = self.audio_file.loc[self.audio_file.slice_file_name == audio_name].iloc[
            0, -2
        ]
        return label


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
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return val_loader

    def test_dataloader(self):
        test_loader = tud.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return test_loader


class UrbanSoundDataModule(pl.LightningDataModule):
    def __init__(
        self,
        annotation_file,
        audio_dir,
        sample_rate=16000,
        num_samples=32000,
        batch_size=128,
        num_workers=0,
        transforms=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transorms = transforms
        self.annotation_file = annotation_file
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.num_samples = num_samples

    def setup(self, stage=None) -> None:
        self.train_dataset = UrbanSoundDataset(
            annotation_file=self.annotation_file,
            audio_dir=self.audio_dir,
            folds=[1, 2, 3, 4, 5, 6, 7, 8],
            target_sample_rate=self.sample_rate,
            num_samples=self.num_samples,
        )
        self.val_dataset = UrbanSoundDataset(
            annotation_file=self.annotation_file,
            audio_dir=self.audio_dir,
            folds=9,
            target_sample_rate=self.sample_rate,
            num_samples=self.num_samples,
        )
        self.test_dataset = UrbanSoundDataset(
            annotation_file=self.annotation_file,
            audio_dir=self.audio_dir,
            folds=10,
            target_sample_rate=self.sample_rate,
            num_samples=self.num_samples,
        )

    def train_dataloader(self):
        train_loader = tud.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = tud.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return val_loader

    def test_dataloader(self):
        test_loader = tud.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return test_loader
