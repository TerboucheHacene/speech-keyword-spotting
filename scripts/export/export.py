import os
import torch


from keyword_detector.cli_args.args import parse_args
from keyword_detector.models import METHODS
from keyword_detector.models.pl_modules import SpeechLightningModel

PROJECT_NAME = "speech-keyword-spotting"
WANDB_PATH = "artifacts/wandb/"
RESULTS_DIR = "artifacts/results/"


def get_onnx_trace(args) -> str:

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
    SAMPLE_RATE = 16000
    CLIP_DURATION = 1
    dummy_input = torch.zeros([1, (SAMPLE_RATE * CLIP_DURATION)])

    torch.onnx.export(
        model=lightning_model.model,
        args=dummy_input,
        f=os.path.join(RESULTS_DIR, args.checkpoint_dir) + "model.onnx",
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        # whether to execute constant folding for optimization
        output_names=["output"],  # the model's output names
        input_names=["input"],  # the model's input names
        dynamic_axes={
            "input": {0: "batch_size"},
            # variable length axes
            "output": {0: "batch_size"},
        },
    )


if __name__ == "__main__":
    args = parse_args()
    print("Loaded arguments:")
    get_onnx_trace(args)
