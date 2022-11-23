# Speech Keyword Spotting
This repo implements a speech keyword spotting system. 

* Build and train a Custom Model based on a pretrained Wav2Vec2 model to detect the keyword based on raw audio
* Implement the evaluation and the experiment tracking using Weights & Biases
* Export the model to TorchScript as well as to ONNX formats.
* Deploy the model as a REST API using FastAPI
* Containerize the code using Docker and Use NGINX as a web proxy server.


## How to use
* Clone this repo
* Install [poetry](https://python-poetry.org/docs/#installation), the ultimate tool for dependency management and packaging in Python, and then install the virtual environment:

        poetry install

* To train the model, execute the following command, where you can change the parameters of the training in the CLI:

        source bash\train.sh

* You should sign in your [Weights & Biases](https://wandb.ai/site) account and log in to track the training/validation metrics. 
* To test a trained model, use:

        source bash\test.sh

* To export the model to ONNX:

        source bash\export.sh

## Dataset

The repo uses the [Speech Commands dataset](https://pytorch.org/audio/stable/generated/torchaudio.datasets.SPEECHCOMMANDS.html#torchaudio.datasets.SPEECHCOMMANDS). It was proposed in a paper entitled: **Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition** by [Pete Warden](https://arxiv.org/abs/1804.03209).

> Describes an audio dataset of spoken words designed to help train and evaluate keyword spotting systems. Discusses why this task is an interesting challenge, and why it requires a specialized dataset that is different from conventional datasets used for automatic speech recognition of full sentences. Suggests a methodology for reproducible and comparable accuracy metrics for this task. Describes how the data was collected and verified, what it contains, previous versions and properties. Concludes by reporting baseline results of models trained on this dataset.
## Modeling
This work uses a pretrained model called wav2vec2, it consists of a Transformer Network trained in a self-supervised manner.

> wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations \
Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli \
https://arxiv.org/abs/2006.11477\
**Abstract** We show for the first time that learning powerful representations from speech audio alone followed by fine-tuning on transcribed speech can outperform the best semi-supervised methods while being conceptually simpler. wav2vec 2.0 masks the speech input in the latent space and solves a contrastive task defined over a quantization of the latent representations which are jointly learned. Experiments using all labeled data of Librispeech achieve 1.8/3.3 WER on the clean/other test sets. When lowering the amount of labeled data to one hour, wav2vec 2.0 outperforms the previous state of the art on the 100 hour subset while using 100 times less labeled data. Using just ten minutes of labeled data and pre-training on 53k hours of unlabeled data still achieves 4.8/8.2 WER. This demonstrates the feasibility of speech recognition with limited amounts of labeled data.

## Deployment
We deploy the model as an HTTP endpoint using FastAPI, and then dockerize the code in a docker image. We also use an NGNIX image as a proxy server. All the code related to the deployment can be found in this [folder](scripts/deploy).

* To build the server image:

        docker build . -f scripts/deploy/fastapi/Dockerfile -t kws

*  To run the model locally:

        docker container run -d -p 900:80 --name myapp kws

* You can send a request to the server:

        python scripts/deploy/request.py

* To deploy the app using docker compose:

        docker-compose up --build

 