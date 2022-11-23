import argparse
import os

import torch

from keyword_detector.cli_args.args import parse_args
from keyword_detector.models import METHODS
from keyword_detector.models.pl_modules import SpeechLightningModel

PROJECT_NAME = "speech-keyword-spotting"
WANDB_PATH = "artifacts/wandb/"
RESULTS_DIR = "artifacts/results/"
SAMPLE_RATE = 16000
CLIP_DURATION = 1
OPSET_VERSION = 13


def get_onnx_trace(args: argparse.Namespace) -> str:

    # Init model
    ModelConfig = METHODS[args.method]
    model = ModelConfig(num_classes=args.num_classes)
    model_path = os.path.join(RESULTS_DIR, args.checkpoint_dir) + "last.ckpt"

    lightning_model = SpeechLightningModel.load_from_checkpoint(
        model_path,
        pt_model=model,
        json_path="",
        **vars(args),
    )
    # define dummy input
    dummy_input = torch.zeros([1, (SAMPLE_RATE * CLIP_DURATION)])
    # get onnx trace
    onnx_path = os.path.join(RESULTS_DIR, args.checkpoint_dir, "model.onnx")

    torch.onnx.export(
        # model being run
        model=lightning_model.model,
        # model input (or a tuple for multiple inputs)
        args=dummy_input,
        # where to save the model (can be a file or file-like object)
        f=onnx_path,
        # wether to export the parameters with constant names
        export_params=True,
        # the ONNX version to export the model to
        opset_version=OPSET_VERSION,
        # do constant folding for optimization
        do_constant_folding=True,
        output_names=["output"],  # the model's output names
        input_names=["input"],  # the model's input names
        # dynamic axes for variable length inputs
        dynamic_axes={
            "input": {0: "batch_size"},
            # variable length axes
            "output": {0: "batch_size"},
        },
    )
    return onnx_path


if __name__ == "__main__":
    args = parse_args()
    print("Loaded arguments:")
    get_onnx_trace(args)
