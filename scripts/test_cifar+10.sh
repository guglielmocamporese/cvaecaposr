#!/bin/bash

python main.py \
    --data_base_path "./data" \
    --dataset "cifar+10" \
    --val_ratio 0.2 \
    --seed 1234 \
    --batch_size 32 \
    --split_num 0 \
    --z_dim 128 \
    --lr 5e-5 \
    --t_mu_shift 10.0 \
    --t_var_scale 0.01 \
    --alpha 1.0 \
    --beta 0.01 \
    --margin 10.0 \
    --in_dim_caps 16 \
    --out_dim_caps 32 \
    --checkpoint "checkpoints/cifar+10.ckpt" \
    --mode "test"
