from abc import ABC, abstractmethod, abstractproperty
from argparse import ArgumentParser
from typing import Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, hub

MODEL_URLS = {
    "vggish": "https://github.com/harritaylor/torchvggish/"
    "releases/download/v0.1/vggish-10086976.pth",
}


class BaseAudioFrameEncoder(ABC, nn.Module):
    """Abstract class to encode a single audio frame"""

    def __init__(self) -> None:
        super().__init__()

    @abstractproperty
    def feature_dim(self):
        pass

    @abstractmethod
    def forward(self, input: Tensor) -> Tensor:
        pass


class VGG(nn.Module):
    """The VGG network definition, composed of a features extraction module, followed by
    an MLP model (3 FC-ReLU layers)"""

    def __init__(self) -> None:
        super(VGG, self).__init__()
        TYPE_LAYERS: List[Union[int, str]] = [
            64,
            "M",
            128,
            "M",
            256,
            256,
            "M",
            512,
            512,
        ]
        self.features = self.make_layers(type_layers=TYPE_LAYERS)
        self.embeddings = nn.Sequential(
            nn.Linear(512 * 4 * 6, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 128),
            nn.ReLU(True),
        )
        self.feature_dim = 128

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        # Transpose the output from features to remain compatible with vggish embeddings
        x = torch.transpose(x, 1, 3)
        x = torch.transpose(x, 1, 2)
        x = x.contiguous()
        x = x.view(x.size(0), -1)

        return self.embeddings(x)

    @staticmethod
    def make_layers(
        type_layers: List[Union[int, str]], in_channels: Union[int, str] = 1
    ) -> nn.Sequential:
        layers: List[nn.Module] = []
        for v in type_layers:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = int(v)
        layers += [nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))]
        return nn.Sequential(*layers)


class VGGish(VGG):
    """An implementation of the VGG network (Conv layers + MLP), called VGGish
    with the possibility to load pretrained weights.

    Parameters
    ----------
    pretrained : bool, optional
        If True, load the pretrained weights on AudioSet dataset, by default True
    urls : Dict[str, str], optional
        url used to load the dictionary state, by default MODEL_URLS
    progress : bool, optional
        if True, show the progress bar while downloading the weights, by default True
    """

    def __init__(
        self,
        pretrained: bool = True,
        urls: Dict[str, str] = MODEL_URLS,
        progress: bool = True,
    ):

        super().__init__()
        if pretrained:
            state_dict = hub.load_state_dict_from_url(urls["vggish"], progress=progress)
            super().load_state_dict(state_dict)

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass of the VGGish network

        Parameters
        ----------
        input : Tensor
            a tensor of shape (B, 1, num_frames, num_mel_bins) for a batch of
            spectrograms

        Returns
        -------
        Tensor
            a tensor of shape (B, 128) for a batch of embeddings
        """

        features = super().forward(input)
        return features


class VGGishWithoutMLP(nn.Module):

    """A slightly different implementation of the VGGish network. Here we remove the
    last fully layers to reduce the complexity of the model.

    """

    def __init__(
        self,
        pretrained: bool = True,
        urls: Dict[str, str] = MODEL_URLS,
        progress: bool = True,
    ):

        super().__init__()
        self.embedding = nn.Sequential(
            self.get_vggish_without_fc(
                pretrained=pretrained, urls=urls, progress=progress
            ),
            nn.AvgPool2d(kernel_size=(6, 4)),
            nn.Flatten(),
        )
        self.feature_dim = 512

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass of the VGGish network

        Parameters
        ----------
        input : Tensor
            a tensor of shape (B, 1, num_frames, num_mel_bins) for a batch of
            spectrograms

        Returns
        -------
        Tensor
            a tensor of shape (B, 512) for a batch of embeddings
        """
        features = self.embedding(input)
        return features

    @staticmethod
    def get_vggish_without_fc(
        pretrained: bool,
        urls: Dict[str, str],
        progress: bool,
    ) -> List[nn.Module]:
        model = VGGish(pretrained=pretrained, urls=urls, progress=progress)
        embeddings = [*model.children()][0]
        return embeddings


class WrapperVGGish(BaseAudioFrameEncoder):
    """Wrapper around the VGGish network to encode a single audio frame

    Parameters
    ----------
    pretrained : bool, optional
        If True, load the pretrained weights on AudioSet dataset, by default True
    urls : Dict[str, str], optional
        url used to load the dictionary state, by default MODEL_URLS
    progress : bool, optional
        if True, show the progress bar while downloading the weights, by default True
    use_mlp : bool, optional
        if True, use the VGGish network with the MLP, by default True
    """

    def __init__(
        self,
        num_classes: int,
        use_mlp: bool = True,
        pretrained: bool = True,
        urls: Dict[str, str] = MODEL_URLS,
        progress: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.use_mlp = use_mlp
        if self.use_mlp:
            self.model = VGGish(pretrained=pretrained, urls=urls, progress=progress)
            self._feature_dim = 128
        else:
            self.model = VGGishWithoutMLP(
                pretrained=pretrained, urls=urls, progress=progress
            )
            self._feature_dim = 512

        self.classification_head = nn.Linear(
            self._feature_dim,
            num_classes,
        )

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--num_classes", type=int, default=35, help="number of classes"
        )
        parser.add_argument(
            "--pretrained", type=bool, default=True, help="load pretrained weights"
        )
        parser.add_argument(
            "--use_mlp",
            type=bool,
            default=True,
            help="if True, use the VGGish network with the MLP",
        )
        return parser

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def forward(self, input: Tensor) -> Tensor:
        """Get the features for the input spectrograms or sequence of spectrograms"""
        features = self.model(input)
        logits = self.classification_head(features)
        return logits
