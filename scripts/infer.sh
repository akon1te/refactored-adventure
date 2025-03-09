#!/bin/bash

python ./src/inference.py \
    --model_path "ckpt/checkpoint-arcane-best" \
    --init_image "init-path" \
    --prompt "arcane style, detailed, high quality, League of Legends" \
    --strength 0.75 \
    --num_inference_steps 50 \
    --output "./data/output_imgs" \
