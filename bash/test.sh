#!/bin/bash



poetry run python scripts/test.py \
    --batch_size 128 \
    --checkpoint_dir "custom_wav2vec2_2022-11-21_21-34-03/" \

