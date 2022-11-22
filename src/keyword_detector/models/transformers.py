from argparse import ArgumentParser

import torch
import torch.nn as nn
from transformers import (
    HubertForSequenceClassification,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
)


class Wav2Vec2AudioModel(nn.Module):
    """Define the Wav2Vec2 model. It is a pre-trained model from the HuggingFace library.
    The model is a sequence classification model, which means that it takes a
    1-second-long audio array (sampled at sample_rate=1600 in mono) and outputs the
    logits of an audio keyword. The vector can then passed through a softmax function
    to get the probability distribution over the classes.

    Parameters
    ----------
    num_classes : int
        Number of classes in the dataset. Default is 35.
    """

    def __init__(
        self, num_classes: int = 35, sampling_rate: int = 16000, **kwargs
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.sampling_rate = sampling_rate
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "superb/wav2vec2-base-superb-er",
            feature_size=1,
            sampling_rate=self.sampling_rate,
        )
        self.wav2vec2_model = Wav2Vec2ForSequenceClassification.from_pretrained(
            "superb/wav2vec2-base-superb-er",
            num_labels=self.num_classes,
            ignore_mismatched_sizes=True,
        )

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("Wav2Vec2AudioModel")
        parser.add_argument(
            "--num_classes", type=int, default=35, help="number of classes"
        )
        parser.add_argument("--sampling_rate", type=int, default=16000)
        return parent_parser

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        # x = x.squeeze()
        inputs = self.feature_extractor(
            x,
            return_tensors="pt",
            sampling_rate=self.sampling_rate,
            max_length=self.sampling_rate,
            return_attention_mask=False,
        )
        inputs = inputs.input_values.squeeze(1).to(x.device)
        logits = self.wav2vec2_model(inputs).logits
        return logits


class CustomWav2Vec2AudioModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 35,
        sampling_rate: int = 16000,
        freeze: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.sampling_rate = sampling_rate
        self.freeze = freeze
        # layers
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "superb/wav2vec2-base-superb-er",
            feature_size=1,
            sampling_rate=self.sampling_rate,
        )
        W2V2_OUTPUT_SIZE = 768
        # layers
        self.w2v2_model = Wav2Vec2ForSequenceClassification.from_pretrained(
            "superb/wav2vec2-base-superb-er"
        )
        self.classifier = torch.nn.Linear(W2V2_OUTPUT_SIZE, self.num_classes)

        if freeze:
            self.freeze()

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("CustomWav2Vec2")
        parser.add_argument(
            "--num_classes", type=int, default=35, help="number of classes"
        )
        parser.add_argument("--sampling_rate", type=int, default=16000)
        parser.add_argument("--freeze", type=bool, default=False)
        return parent_parser

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # inputs = self.feature_extractor(
        #     x,
        #     return_tensors="pt",
        #     sampling_rate=self.sampling_rate,
        #     max_length=self.sampling_rate,
        #     return_attention_mask=False,
        # )
        # inputs = inputs.input_values.squeeze(1).to(x.device)
        features = self.w2v2_model.wav2vec2(x).last_hidden_state
        pooled = features.mean(dim=1)
        logits = self.classifier(pooled)
        return logits

    def freeze(self) -> None:
        # freeze the wav2vec2 model
        for param in self.w2v2_model.parameters():
            param.requires_grad = False


class HubertAudioModel(nn.Module):
    """This model takes in a batch of audio files and returns a batch of predictions.
    The predictions are the probability that the audio file contains a keyword.

    Parameters
    ----------
    num_classes : int
        The number of classes to predict. This is the number of keywords.
    """

    def __init__(
        self, num_classes: int = 35, sampling_rate: int = 16000, **kwargs
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.sampling_rate = sampling_rate
        # layers
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "superb/wav2vec2-base-superb-er", sampling_rate=self.sampling_rate
        )
        self.hubert_model = HubertForSequenceClassification.from_pretrained(
            "superb/hubert-large-superb-er",
            num_labels=self.num_classes,
            ignore_mismatched_sizes=True,
        )

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("HubertAudioModel")
        parser.add_argument(
            "--num_classes", type=int, default=35, help="number of classes"
        )
        parser.add_argument("--sampling_rate", type=int, default=16000)
        return parent_parser

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        inputs = self.feature_extractor(
            x,
            return_tensors="pt",
            sampling_rate=self.sampling_rate,
            max_length=self.sampling_rate,
            return_attention_mask=False,
        )
        inputs = inputs.input_values.squeeze(1).to(x.device)
        logits = self.hubert_model(inputs).logits
        return logits


class LightHubertAudioModel(nn.Module):
    """This model takes in a batch of audio files and returns a batch of predictions.
    The predictions are the probability that the audio file contains a keyword among
    a predefined list of keywords.

    Parameters
    ----------
    num_classes : int
        The number of classes to predict. This is the number of keywords.
    hidden_dim : int
        The number of hidden dimensions in the model.
    """

    def __init__(self, num_classes: int = 35, hidden_dim: int = 128, **kwargs) -> None:
        super().__init__()
        self.num_classes = num_classes

        # layers
        w2v2_model = HubertForSequenceClassification.from_pretrained(
            "superb/hubert-large-superb-er"
        )
        self.extractor = w2v2_model.hubert.feature_extractor
        self.bn_in = torch.nn.BatchNorm1d(512)
        self.conv = torch.nn.Conv1d(512, hidden_dim, kernel_size=7, stride=3, bias=False)
        self.bn = torch.nn.BatchNorm1d(hidden_dim)
        self.relu = torch.nn.ReLU(inplace=True)
        self.downsample = torch.nn.AvgPool1d(4, 4)
        self.conv_out = torch.nn.Conv1d(
            hidden_dim, self.num_classes, kernel_size=7, stride=15
        )

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("LightHubertAudioModel")
        parser.add_argument(
            "--num_classes", type=int, default=35, help="number of classes"
        )
        parser.add_argument("--hidden_dim", type=int, default=128)
        return parent_parser

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extractor(x)
        x = self.bn_in(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.downsample(x)
        x = self.conv_out(x)
        return x
