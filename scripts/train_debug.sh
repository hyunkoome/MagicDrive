#!/bin/bash

accelerate launch --config_file /home/hyunkoo/DATA/ssd8tb/Journal/MagicDrive/configs/accelerator/accelerate_config_2gpu.yaml tools/train.py \
+exp=224x400 runner=debug runner.validation_before_run=true
