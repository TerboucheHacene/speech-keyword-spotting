import os
from datetime import datetime

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from keyword_detector.cli_args.args import parse_args
from keyword_detector.data.data_modules import SpeechDataModule
from keyword_detector.models import METHODS
from keyword_detector.models.pl_modules import SpeechLightningModel

PROJECT_NAME = "speech-keyword-spotting"
WANDB_PATH = "artifacts/wandb/"
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
    model = ModelConfig(num_classes=args.num_classes)
    model_path = os.path.join(RESULTS_DIR, args.checkpoint_dir) + "last.ckpt"

    lightning_model = SpeechLightningModel.load_from_checkpoint(
        model_path,
        pt_model=model,
        json_path=experiment_dir,
        **vars(args),
    )

    # Init data loader
    data_module = SpeechDataModule(**vars(args))

    # Init trainer
    trainer = Trainer.from_argparse_args(
        args,
        gpus=1,
        logger=experiment,
        log_every_n_steps=1,
    )

    # Test
    trainer.test(lightning_model, data_module)


if __name__ == "__main__":
    args = parse_args()
    print("Loaded arguments:")
    train(args)
