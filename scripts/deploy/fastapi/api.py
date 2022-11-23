import io
import os
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
from fastapi import Depends, FastAPI, File, Request, UploadFile, status
from onnxruntime import InferenceSession

from keyword_detector.data.data_utils import index_to_label

from .schemas import AudioMetaData, Settings, SpeechKeyWord

# Define application
app = FastAPI(
    title="Keyword Spotting API",
    description="API for keyword spotting, based on wav2vec2.0",
    version="0.1",
)

settings = Settings()


class KeywordSpottingModel:
    model: Optional[InferenceSession] = None

    def load_model(self):
        path = os.path.join(settings.model_path, settings.model_name)
        print("Loading model from ...", path)
        # Create session
        self.model = InferenceSession(path)

    async def predict(
        self,
        params: AudioMetaData = Depends(),
        waveform: UploadFile = File(description="audio file"),
    ) -> SpeechKeyWord:
        print(params, waveform)
        if not self.model:
            raise RuntimeError
        # Read audio file
        audio_bytes = waveform.file.read()
        # Convert to io.BytesIO
        audio = io.BytesIO(audio_bytes)
        # Convert to numpy array
        signal, sr = sf.read(
            audio,
            dtype="float32",
            start=int(params.offset * params.sr),
            stop=int(params.offset + params.duration * params.sr),
        )
        # Resample
        if sr != params.sr:
            signal = librosa.resample(signal, sr, params.sr)
        # Create input
        waveform_batch = signal.reshape(1, -1).astype(np.float32)
        input = {"input": waveform_batch}
        # Run inference
        output = self.model.run(["output"], input)
        # Get results
        index = output[0].argmax(axis=1).item()
        prediction = {"key_word": index_to_label(index)}
        return SpeechKeyWord(**prediction)


classification_model = KeywordSpottingModel()


@app.on_event("startup")
def load_artifacts():
    classification_model.load_model()
    print("Ready for inference!")


@app.get("/", tags=["General"], status_code=status.HTTP_200_OK)
def _index(request: Request):
    """Health check."""
    response = {
        "data": "Everything is working as expected",
    }
    return response


@app.post("/predict", tags=["Prediction"], status_code=status.HTTP_200_OK)
async def _predict(
    output: SpeechKeyWord = Depends(classification_model.predict),
) -> SpeechKeyWord:
    # Predict
    return output
