from .cnn import M5
from .vgg import WrapperVGGish
from .transformers import Wav2Vec2AudioModel, HubertAudioModel, LightHubertAudioModel


METHODS = {
    "vggish": WrapperVGGish,
    "m5": M5,
    "wav2vec2": Wav2Vec2AudioModel,
    "hubert": HubertAudioModel,
    "light_hubert": LightHubertAudioModel,
}
