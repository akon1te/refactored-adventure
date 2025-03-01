#!/bin/bash

accelerate launch finetune.py \
    --image_dir "/путь/к/изображениям" \
    --batch_size 4 \
    --num_epochs 3 \
    --learning_rate 1e-5 \
    --prompt "arcane style" \
    --save_path "fine-tuned-sd-controlnet" \