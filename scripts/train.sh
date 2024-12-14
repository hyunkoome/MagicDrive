#!/bin/bash

accelerate launch --config_file /home/hyunkoo/DATA/ssd8tb/Journal/MagicDrive/configs/accelerator/accelerate_config_2gpu.yaml tools/train.py \
+exp=224x400 runner=2gpus
