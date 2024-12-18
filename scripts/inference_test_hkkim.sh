#!/bin/bash

#python tools/testhkkim.py resume_from_checkpoint=./pretrained/SDv1.5mv-rawbox_2023-09-07_18-39_224x400

#python tools/testhkkim.py resume_from_checkpoint=/home/hyunkoo/DATA/ssd8tb/Journal/MagicDrive/pretrained/SDv1.5mv-rawbox_2023-09-07_18-39_224x400

python tools/inference_test_hkkim.py resume_from_checkpoint=/home/hyunkoo/DATA/ssd8tb/Journal/MagicDrive/magicdrive-log/model_convert/SDv1.5mv-rawbox_2024-12-17_23-16_224x400
