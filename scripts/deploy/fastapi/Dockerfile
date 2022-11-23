FROM python:3.8.5

RUN mkdir MyApp
WORKDIR /MyApp

ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 
ENV PYTHONPATH=${PYTHONPATH}:${PWD}

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
# Install Poetry
RUN pip install --upgrade pip
RUN pip install poetry==1.2.1
RUN poetry config virtualenvs.create false --local 

COPY pyproject.toml .
RUN poetry install --no-root

# install the project
COPY src src
RUN poetry install --only-root

# copy needed scripts and files
COPY scripts/deploy/fastapi/ deploy/
COPY artifacts/results/custom_wav2vec2_2022-11-21_21-34-03/model.onnx artifacts/results/custom_wav2vec2_2022-11-21_21-34-03/model.onnx

EXPOSE 900

CMD ["uvicorn", "deploy.api:app", "--host" , "0.0.0.0", "--port",  "900", "--root-path", "/api"]
