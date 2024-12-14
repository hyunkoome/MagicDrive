#!/bin/bash

accelerate launch --main_process_port 29501 --config_file /home/hyunkoo/DATA/ssd8tb/Journal/MagicDrive/configs/accelerator/accelerate_config_2gpu.yaml python demo/interactive_gui.py

# 아래로 설치해야된다고 함
#    pip install --upgrade pydantic==2.8.2
#    pip install --upgrade pydantic-core==2.20.1
#    pip install --upgrade fastapi==0.112.4