accelerate launch --config_file /home/hyunkoo/DATA/ssd8tb/Journal/MagicDrive/configs/accelerator/accelerate_config_1gpu.yaml \
tools/save_pytorch_model_from_accelerate_checkpoint.py \
resume_from_checkpoint=/home/hyunkoo/DATA/ssd8tb/Journal/MagicDrive/magicdrive-log/SDv1.5mv-rawbox_2024-12-13_21-38_224x400/checkpoint-160000 \
+exp=224x400 runner=2gpus