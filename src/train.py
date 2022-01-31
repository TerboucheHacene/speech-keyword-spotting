# import comet_ml at the top of your file
from comet_ml import Experiment
from pytorch_lightning.loggers import CometLogger
import argparse
import torchaudio

from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DDPPlugin

from models.models import ClassificationModel, M5
from data.data import SpeechDataModule


def parse_args():
    parser = argparse.ArgumentParser()
    # CLI args
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--freeze_bert_layer", type=eval, default=False)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--momentum", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_seq_length", type=int, default=128)
    # Container environment
    parser.add_argument(
        "--train_data", type=str, default="./output_data/sentiment/train.tsv"
    )
    parser.add_argument(
        "--validation_data", type=str, default="./output_data/sentiment/train.tsv"
    )
    parser.add_argument("--output_dir", type=str, default="./results/")
    parser.add_argument("--num_gpus", type=int, default=0)
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
    )

    model = ClassificationModel(
        model=M5(),
        learning_rate=args.learning_rate,
        transforms=get_transform(16000),
    )

    data_module = SpeechDataModule(batch_size=args.batch_size, num_workers=0)

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator=None,
        num_sanity_val_steps=-1,
        val_check_interval=0.5,
        terminate_on_nan=True,
        logger=experiment,
        log_every_n_steps=1,
    )

    trainer.fit(model, data_module)


if __name__ == "__main__":
    args = parse_args()
    print("Loaded arguments:")
    train(args)
