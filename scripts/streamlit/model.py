import os
from typing import List

import torch


class Model:
    def __init__(self, model_path: str) -> None:
        self.model = torch.jit.load(model_path, map_location=torch.device("cpu"))
        self.model.eval()
        self.buffer: List[torch.Tensor] = []

    def predict(self, input: torch.Tensor) -> torch.Tensor:
        return self.model(input)
