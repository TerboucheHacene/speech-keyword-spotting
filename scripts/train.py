# import comet_ml at the top of your file
import os
from datetime import datetime

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from keyword_detector.cli_args.args import parse_args
from keyword_detector.data.data_modules import SpeechDataModule
from keyword_detector.models import METHODS
from keyword_detector.models.pl_modules import SpeechLightningModel

PROJECT_NAME = "speech-keyword-spotting"
WANDB_PATH = "artifacts/wandb/"
MIN_DELTA = 0.01
METRIC_TO_MONITOR = "val/f1"
MODEL_DIR = "artifacts/models/"
RESULTS_DIR = "artifacts/results/"


def train(args):
    experiment_name = f"{args.method}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    experiment_dir = os.path.join(RESULTS_DIR, experiment_name)

    os.makedirs(experiment_dir, exist_ok=True)

    # init logger
    experiment = WandbLogger(
        project=PROJECT_NAME,
        save_dir=WANDB_PATH,
        name=experiment_name,
    )

    # Init model
    ModelConfig = METHODS[args.method]
    pt_model = ModelConfig(num_classes=args.num_classes)
    lightning_model = SpeechLightningModel(
        pt_model=pt_model,
        json_path=experiment_dir,
        **vars(args),
    )

    # Init data loader
    data_module = SpeechDataModule(**vars(args))

    # Init callbacks
    early_stopping = EarlyStopping(
        monitor=METRIC_TO_MONITOR,
        patience=10,
        mode="max",
        min_delta=MIN_DELTA,
        check_on_train_epoch_end=False,
        verbose=True,
    )
    model_checkpoint = ModelCheckpoint(
        dirpath=MODEL_DIR + f"{experiment_name}/",
        filename="model_{epoch}",
        monitor=METRIC_TO_MONITOR,
        mode="max",
        save_top_k=1,
        verbose=True,
        save_last=True,
    )
    callbacks = [early_stopping, model_checkpoint]

    # Init trainer
    trainer = Trainer.from_argparse_args(
        args,
        gpus=1,
        logger=experiment,
        log_every_n_steps=1,
        detect_anomaly=True,
        callbacks=callbacks,
        # limit_train_batches=0.1,
        # fast_dev_run=True,
    )

    # Train
    trainer.fit(lightning_model, data_module)

    # Test
    trainer.test(lightning_model, data_module)


if __name__ == "__main__":
    args = parse_args()
    print("Loaded arguments:")
    train(args)
