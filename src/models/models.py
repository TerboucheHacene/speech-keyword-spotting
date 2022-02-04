import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch import hub


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


class VGG(nn.Module):

    """The VGG network definition, composed of a features extraction module, followed by
    an MLP model (3 FC-ReLU layers)
    Attributes
    ----------
    feature_dim : int
        dimension size of the output feature vector
    features : nn.Module
        neural network that extractes features
    """

    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.embeddings = nn.Sequential(
            nn.Linear(512 * 4 * 6, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 128),
            nn.ReLU(True),
        )

        self.feature_dim = 128

    def forward(self, x):
        x = self.features(x)
        # Transpose the output from features to
        # remain compatible with vggish embeddings
        x = torch.transpose(x, 1, 3)
        x = torch.transpose(x, 1, 2)
        x = x.contiguous()
        x = x.view(x.size(0), -1)

        return self.embeddings(x)


def make_layers():
    layers = []
    in_channels = 1
    for v in [64, "M", 128, "M", 256, 256, "M", 512, 512]:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    layers += [nn.AvgPool2d(kernel_size=(2, 4), stride=(2, 2))]
    return nn.Sequential(*layers)


def _vgg():
    return VGG(make_layers())


model_urls = {
    "vggish": "https://github.com/harritaylor/torchvggish/"
    "releases/download/v0.1/vggish-10086976.pth",
    "pca": "https://github.com/harritaylor/torchvggish/"
    "releases/download/v0.1/vggish_pca_params-970ea276.pth",
}


class VGGishV1(VGG, nn.Module):
    """An implementation of the VGG network (features extraction + MLP), called VGGishV1
    with the possiblity to load pretrained weights. The model can take as input either a
    frame of spectrogram of a sequence of them (set the parameters 'seq' to True)
    Parameters
    ----------
    pretrained : bool, optional
        load pretrained weights or not
    """

    def __init__(self, pretrained=True, urls=model_urls, progress=True):
        super().__init__(make_layers())
        if pretrained:
            state_dict = hub.load_state_dict_from_url(urls["vggish"], progress=progress)
            super().load_state_dict(state_dict)

    def forward(self, x):
        y = VGG.forward(self, x)
        return y


def get_vggish_without_fc(pretrained=True):
    model = VGGishV1(pretrained=pretrained)
    embeddings = [*model.children()][0]
    return embeddings


class VGGishV2(nn.Module):

    """This implementation contains only the VGG backbone (without the FC layers)"""

    def __init__(self, pretrained=True):
        super().__init__()
        self.embedding = nn.Sequential(
            get_vggish_without_fc(pretrained),
            nn.AvgPool2d(kernel_size=(6, 4)),
            nn.Flatten(),
        )
        self.feature_dim = 512

    def forward(self, x):
        y = self.embedding(x)
        return y


class VGGish(nn.Module):

    """A wrapper module to choose betwenn version 01(with MLP) and version 02 (without MLP)
    Attributes
    ----------
    use_fc : bool, optional
        whether to use the FC layers or not, if True, then use version 02
    """

    def __init__(self, num_classes, use_fc=True, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        if use_fc:
            self.model = VGGishV1(pretrained)
            self.feature_dim = 128
        else:
            self.model = VGGishV2(pretrained)
            self.feature_dim = 512
        self.fc = nn.Linear(self.feature_dim, self.num_classes)

    def forward(self, x):
        y = self.model(x)
        y = F.log_softmax(self.fc(y), dim=-1)
        return y
