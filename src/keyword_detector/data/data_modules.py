from argparse import ArgumentParser
from typing import Callable

import pytorch_lightning as pl
import torch.utils.data as tud

from .data_utils import collate_fn, collate_fn_spec
from .datasets import SubsetSC, UrbanSoundDataset


class SpeechDataModule(pl.LightningDataModule):
    """LightningDataModule for Speech Commands Dataset

    Parameters
    ----------
    batch_size : int, optional
        Batch size, by default 128
    num_workers : int, optional
        Number of workers, by default 0
    transforms : callable, optional
        A function/transform that takes in a waveform and returns a transformed
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        transforms: Callable,
        **kwargs,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms = transforms

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("SpeechDataModule")
        parser.add_argument("--data_dir", type=str, default="artifacts/")
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--num_workers", type=int, default=0)
        parser.set_defaults(transforms=None)
        return parent_parser

    def setup(self, stage=None) -> None:
        self.train_dataset = SubsetSC(subset="training")
        self.val_dataset = SubsetSC(subset="validation")
        self.test_dataset = SubsetSC(subset="testing")
        self.labels = sorted(list(set(datapoint[2] for datapoint in self.train_dataset)))

    def train_dataloader(self):
        """Returns the training dataloader"""
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
        """Returns the validation dataloader"""
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
        """Returns the test dataloader"""
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
    """LightningDataModule for UrbanSound8K Dataset

    Parameters
    ----------
    annotation_file : str
        Path to the annotation file
    audio_dir : str
        Path to the audio directory
    sample_rate : int, optional
        Target sample rate, by default 16000
    num_samples : int, optional
        Number of samples, by default 16000
    batch_size : int, optional
        Batch size, by default 128
    num_workers : int, optional
        Number of workers, by default 0
    transforms : callable, optional
        A function/transform that takes in a waveform and returns a transformed
    """

    def __init__(
        self,
        annotation_file: str,
        audio_dir: str,
        sample_rate: int = 16000,
        num_samples: int = 16000,
        batch_size: int = 128,
        num_workers: int = 0,
        transforms: Callable = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms = transforms
        self.annotation_file = annotation_file
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.num_samples = num_samples

    def setup(self, stage=None) -> None:
        self.train_dataset = UrbanSoundDataset(
            annotation_file=self.annotation_file,
            audio_dir=self.audio_dir,
            folds=[1, 2, 3, 4, 5, 6, 7, 8],
            transforms=self.transforms,
            target_sample_rate=self.sample_rate,
            num_samples=self.num_samples,
        )
        self.val_dataset = UrbanSoundDataset(
            annotation_file=self.annotation_file,
            audio_dir=self.audio_dir,
            folds=9,
            transforms=self.transforms,
            target_sample_rate=self.sample_rate,
            num_samples=self.num_samples,
        )
        self.test_dataset = UrbanSoundDataset(
            annotation_file=self.annotation_file,
            audio_dir=self.audio_dir,
            folds=10,
            transforms=self.transforms,
            target_sample_rate=self.sample_rate,
            num_samples=self.num_samples,
        )

    def train_dataloader(self):
        train_loader = tud.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn_spec,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = tud.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn_spec,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return val_loader

    def test_dataloader(self):
        test_loader = tud.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn_spec,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return test_loader
