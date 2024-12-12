#!/bin/bash

accelerate launch --main_process_port 29501 --config_file /home/hyunkoo/DATA/ssd8tb/Journal/MagicDrive/configs/accelerator/accelerate_config_2gpu.yaml python demo/interactive_gui.py