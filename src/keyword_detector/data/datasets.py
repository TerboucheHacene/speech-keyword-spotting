import torch
import torchaudio
import torch.utils.data as tud
from torchaudio.datasets import SPEECHCOMMANDS
import os
import glob
import pandas as pd
from typing import Callable, Union, List


class SubsetSC(SPEECHCOMMANDS):
    """Speech Commands Dataset

    Parameters
    ----------
    root : str
        Path to the directory where the dataset is found or downloaded.
    subset : str, optional
        Subset of the dataset to consider, by default None
        Options: "training", "validation", "testing"
    download : bool, optional
        Whether to download the dataset if it is not found at root path, by default False

    """

    def __init__(self, root="artifacts/", subset: str = None, download: bool = False):
        super().__init__(root, download=download)

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


class UrbanSoundDataset(tud.Dataset):
    """UrbanSound8K Dataset

    Parameters
    ----------
    annotation_file : str
        Path to the annotation file
    audio_dir : str
        Path to the audio directory
    folds : Union[int, List[int]]
        Fold number to use for training/validation/testing
    transforms : callable, optional
        A function/transform that takes in a waveform and returns a transformed
        version, by default None
    target_sample_rate : int, optional
        Target sample rate, by default 16000
    num_samples : int, optional
        Number of samples to cut or pad, by default 16000
    """

    def __init__(
        self,
        annotation_file: str,
        audio_dir: str,
        folds=Union[int, List[int]],
        transforms: Callable = None,
        target_sample_rate: int = 16000,
        num_samples: int = 16000,
    ):
        self.audio_file = pd.read_csv(annotation_file)
        self.folds = folds
        self.audio_paths = glob.glob(audio_dir + "/*" + str(self.folds) + "/*")
        self.num_samples = num_samples
        self.target_sample_rate = target_sample_rate
        self.transforms = transforms

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):

        audio_sample_path = self._get_audio_sample_path(idx)
        label = self._get_audio_sample_label(audio_sample_path)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)

        # signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        if self.transforms is not None:
            signal = self.transforms(signal)
            signal = signal.unfold(-1, 96, 48).unsqueeze(0)
            signal = torch.transpose(signal, 3, 0).squeeze(3)
            # signal = signal[0, :].unsqueeze(0)
            # signal = signal[0:8, :]

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
        return torch.tensor(label)
