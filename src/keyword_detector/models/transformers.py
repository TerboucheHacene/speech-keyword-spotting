from transformers import (
    HubertForSequenceClassification,
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2FeatureExtractor,
)
import torch.nn as nn
import torch


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

    def __init__(self, num_classes: int = 35, sampling_rate: int = 16000):
        super().__init__()
        self.num_classes = num_classes
        self.sampling_rate = sampling_rate
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "superb/wav2vec2-base-superb-ks",
            feature_size=1,
            sampling_rate=self.sampling_rate,
        )
        self.wav2vec2_model = Wav2Vec2ForSequenceClassification.from_pretrained(
            "superb/wav2vec2-base-superb-ks",
            num_labels=self.num_classes,
            ignore_mismatched_sizes=True,
        )

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
        inputs = inputs.input_values.squeeze().to(x.device)
        logits = self.wav2vec2_model(inputs).logits
        return logits


class HubertAudioModel(nn.Module):
    """This model takes in a batch of audio files and returns a batch of predictions.
    The predictions are the probability that the audio file contains a keyword.

    Parameters
    ----------
    num_classes : int
        The number of classes to predict. This is the number of keywords.
    """

    def __init__(self, num_classes: int = 35, sampling_rate: int = 16000):
        super().__init__()
        self.num_classes = num_classes
        self.sampling_rate = sampling_rate
        # layers
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "superb/wav2vec2-base-superb-ks", sampling_rate=self.sampling_rate
        )
        self.hubert_model = HubertForSequenceClassification.from_pretrained(
            "superb/hubert-large-superb-ks",
            num_labels=self.num_classes,
            ignore_mismatched_sizes=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        inputs = self.feature_extractor(
            x,
            return_tensors="pt",
            sampling_rate=self.sampling_rate,
            max_length=self.sampling_rate,
            return_attention_mask=False,
        )
        inputs = inputs.input_values.squeeze().to(x.device)
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

    def __init__(self, num_classes: int = 35, hidden_dim: int = 128):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extractor(x)
        x = self.bn_in(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.downsample(x)
        x = self.conv_out(x)
        return x
