import os
import argparse
from onnxruntime import InferenceSession
import numpy as np


def main(args):
    path = os.path.join("artifacts", "results", args.checkpoint, "model.onnx")
    print("Loading model from ...", path)

    # Create session
    model = InferenceSession(path)
    output_name = model.get_outputs()[0].name
    input_name = model.get_inputs()[0].name

    print("Input name:", input_name)
    print("Output name:", output_name)
    # Create input
    SAMPLE_RATE = 16000
    CHIP_SIZE = 1

    waveform = np.random.randn(1, SAMPLE_RATE * CHIP_SIZE).astype(np.float32)
    dummy = {"input": waveform}
    results = model.run(["output"], dummy)
    print("Done")
    print(results[0].shape)
    print(results[0])
    print(results[0].argmax(axis=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, default="custom_wav2vec2_2022-11-21_21-34-03"
    )
    args = parser.parse_args()

    main(args)
