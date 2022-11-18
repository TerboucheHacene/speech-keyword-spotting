# import comet_ml at the top of your file
from pytorch_lightning.loggers import CometLogger
import argparse
import torchaudio

from pytorch_lightning import Trainer

from keyword_detector.models.vgg import WrapperVGGish
from keyword_detector.models.pl_modules import ClassificationModel
from keyword_detector.data.data_modules import UrbanSoundDataModule


def parse_args():
    parser = argparse.ArgumentParser()
    # CLI args
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=40)
    parser.add_argument("--learning_rate", type=float, default=5e-3)
    parser.add_argument("--momentum", type=float, default=0.5)
    return parser.parse_args()


def get_transform(sample_rate):
    new_sample_rate = 8000
    transform = torchaudio.transforms.Resample(
        orig_freq=sample_rate, new_freq=new_sample_rate
    )
    return transform


def train(args):
    # Create an experiment with your api key
    experiment = CometLogger(
        api_key="F8z2rvZxchPyTT2l1IawCAE7G",
        project_name="speech-keyword-spotting",
        workspace="ihssen",
        save_dir="artifacts/",
    )

    model = ClassificationModel(
        model=WrapperVGGish(num_classes=10, use_fc=True, pretrained=True),
        learning_rate=args.learning_rate,
        transforms=None,
    )

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=400, hop_length=160, n_mels=64
    )
    data_module = UrbanSoundDataModule(
        annotation_file="/raid/home/labuserterbouche/workspace/UrbanSound8K/metadata/UrbanSound8K.csv",
        audio_dir="/raid/home/labuserterbouche/workspace/UrbanSound8K/audio/",
        transforms=mel_spectrogram,
        sample_rate=16000,
        num_samples=16000,
        batch_size=args.batch_size,
        num_workers=8,
    )

    trainer = Trainer(
        max_epochs=args.max_epochs,
        gpus=1,
        logger=experiment,
        log_every_n_steps=50,
    )

    trainer.fit(model, data_module)
    trainer.test(model, data_module)


if __name__ == "__main__":
    args = parse_args()
    print("Loaded arguments:")
    train(args)
