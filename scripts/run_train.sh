#!/bin/bash

python ./src/train.py \
    --learning_rate 0.00001 \
    --batch_size 2 \
    --num_epochs 10 \
    --data_dir "./data/images" \
    --logs_dir "./logs/arcane_finetune" \
    --model_id "stable-diffusion-v1-5/stable-diffusion-v1-5" \
    --prompt "arcane style, detailed, high quality, League of Legends"