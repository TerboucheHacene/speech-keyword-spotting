from .cnn import M5
from .transformers import (CustomWav2Vec2AudioModel, HubertAudioModel,
                           LightHubertAudioModel, Wav2Vec2AudioModel)
from .vgg import WrapperVGGish

METHODS = {
    "vggish": WrapperVGGish,
    "m5": M5,
    "wav2vec2": Wav2Vec2AudioModel,
    "hubert": HubertAudioModel,
    "light_hubert": LightHubertAudioModel,
    "custom_wav2vec2": CustomWav2Vec2AudioModel,
}
