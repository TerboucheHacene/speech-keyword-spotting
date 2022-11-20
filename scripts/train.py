# import comet_ml at the top of your file
from pytorch_lightning.loggers import WandbLogger
import argparse

from pytorch_lightning import Trainer

from keyword_detector.models import METHODS
from keyword_detector.models.pl_modules import SpeechLightningModel
from keyword_detector.data.data_modules import SpeechDataModule
import json


def parse_args():
    parser = argparse.ArgumentParser()
    # CLI args
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=40)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.5)
    parser.add_argument("--num_classes", type=int, default=35)
    return parser.parse_args()


def train(args):

    experiment = WandbLogger(
        project="speech-keyword-spotting",
        save_dir="artifacts/",
    )

    ModelConfig = METHODS["wav2vec2"]
    model = ModelConfig(num_classes=args.num_classes)

    lightning_model = SpeechLightningModel(
        model=model,
        learning_rate=args.learning_rate,
        transforms=None,
        num_classes=args.num_classes,
    )

    data_module = SpeechDataModule(
        batch_size=args.batch_size,
        num_workers=8,
    )
    trainer = Trainer(
        max_epochs=args.max_epochs,
        gpus=1,
        logger=experiment,
        log_every_n_steps=1,
        detect_anomaly=True,
        # limit_train_batches=0.1,
        # fast_dev_run=True,
    )

    trainer.fit(lightning_model, data_module)
    test_results = trainer.test(lightning_model, data_module)
    # save results to json file


if __name__ == "__main__":
    args = parse_args()
    print("Loaded arguments:")
    train(args)
