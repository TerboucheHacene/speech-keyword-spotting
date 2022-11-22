#!/bin/bash



poetry run python scripts/train.py \
    --batch_size 128 \
    --max_epochs 50 \
    --learning_rate 1e-5