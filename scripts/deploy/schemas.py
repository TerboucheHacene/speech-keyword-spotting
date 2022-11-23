from pathlib import Path

from fastapi import Query
from pydantic import BaseModel, BaseSettings, validator

from keyword_detector.data.data_utils import LABELS


class AudioMetaData(BaseModel):
    """Audio file to be processed"""

    sr: int = Query(16000, description="Sampling rate of audio file")
    duration: float = Query(1.0, description="Duration of audio file in seconds")
    offset: float = Query(0.0, description="Offset of audio file in seconds")


class SpeechKeyWord(BaseModel):
    key_word: str

    @validator("key_word")
    def _validate_key_word(cls, key_word):
        if key_word not in LABELS:
            raise ValueError("Invalid key word")
        return key_word


class Settings(BaseSettings):
    model_path: Path = Path("artifacts/results/custom_wav2vec2_2022-11-21_21-34-03/")
    model_name: str = "model.onnx"
