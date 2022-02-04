import torch.utils.data as tud
import pytorch_lightning as pl
from .datasets import SubsetSC, UrbanSoundDataset
from .data_utils import collate_fn


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
        num_samples=16000,
        batch_size=128,
        num_workers=0,
        transforms=None,
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
