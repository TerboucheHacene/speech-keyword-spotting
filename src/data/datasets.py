import torch
import torchaudio
import torch.utils.data as tud
from torchaudio.datasets import SPEECHCOMMANDS
import os
import glob
import pandas as pd


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


class UrbanSoundDataset(tud.Dataset):
    """
    A rapper class for the UrbanSound8K dataset.
    """

    def __init__(
        self,
        annotation_file,
        audio_dir,
        folds,
        transforms=None,
        target_sample_rate=16000,
        num_samples=16000,
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
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        if self.transforms is not None:
            signal = self.transforms(signal)
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
