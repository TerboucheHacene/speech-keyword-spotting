import os
import torch
import argparse

from keyword_detector.cli_args.args import parse_args
from keyword_detector.models import METHODS
from keyword_detector.models.pl_modules import SpeechLightningModel


RESULTS_DIR = "artifacts/results/"
SAMPLE_RATE = 16000
CLIP_DURATION = 1


def get_traced_model(args: argparse.Namespace) -> str:
    ModelConfig = METHODS[args.method]
    model = ModelConfig(**vars(args))
    checkpoint_path = os.path.join(RESULTS_DIR, args.checkpoint_dir) + "last.ckpt"
    lightning_model = SpeechLightningModel.load_from_checkpoint(
        checkpoint_path,
        pt_model=model,
        json_path="",
        **vars(args),
    )
    # define model path
    model_path = os.path.join(RESULTS_DIR, args.checkpoint_dir, "model.pt")
    # define dummy input
    dummy_input = torch.zeros([1, (SAMPLE_RATE * CLIP_DURATION)])
    # get traced model
    with torch.no_grad():
        traced_model = torch.jit.trace(lightning_model.model.cpu().eval(), dummy_input)
    # save traced model
    torch.jit.save(traced_model, model_path)
    return model_path


if __name__ == "__main__":
    args = parse_args()
    get_traced_model(args)
