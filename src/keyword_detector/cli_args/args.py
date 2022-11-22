import argparse

import pytorch_lightning as pl

from keyword_detector.data.data_modules import SpeechDataModule
from keyword_detector.models import METHODS
from keyword_detector.models.pl_modules import SpeechLightningModel


def parse_train_configs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tune", action="store_true", default=False)
    parser.add_argument("--use_checkpointing", action="store_true", default=False)
    parser.add_argument("--save", action="store_true", default=True)
    parser.add_argument("--artifact_path", type=str, default="googlenet_vggish_dataset")
    parser.add_argument("--method", type=str, default="custom_wav2vec2")
    parser.add_argument("--checkpoint_dir", type=str, default="")


def parse_args():
    parser = argparse.ArgumentParser()
    parse_train_configs(parser)

    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()
    parser = METHODS[temp_args.method].add_model_specific_args(parser)

    # add lightning module args
    parser = SpeechLightningModel.add_model_specific_args(parser)

    # add trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # add data args
    parser = SpeechDataModule.add_argparse_args(parser)

    # parse args
    args = parser.parse_args()

    return args
