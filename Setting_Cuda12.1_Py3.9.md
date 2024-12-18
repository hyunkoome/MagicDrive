## 1. Create Conda Env

```shell
conda create -n mdrive39 python==3.9 -y
conda activate mdrive39
```

## 2. Install Python Packages
Download [mmcv-full==1.7.2](https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html) (mmcv_full-1.7.2-cp39-cp39-manylinux1_x86_64.whl) 
And install
```shell
conda activate mdrive39
python -m pip install mmcv_full-1.7.2-cp39-cp39-manylinux1_x86_64.whl 
```
```shell
pip install -r requirements/py39cu12_1/requirements.txt

cd third_party/diffusers
pip install .

cd third_party/bevfusion 
python setup.py develop
```

### When install bevfusion: 
#### [Error] nvcc fatal   : Unsupported gpu architecture 'compute_80'
- Now, the lastest bevfusion, even can be installed cuda12.1.

```python
if (torch.cuda.is_available() and torch.version.cuda is not None) or os.getenv("FORCE_CUDA", "0") == "1":
    define_macros += [("WITH_CUDA", None)]
    extension = CUDAExtension
    extra_compile_args["nvcc"] = extra_args + [
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
        "-gencode=arch=compute_70,code=sm_70",
        "-gencode=arch=compute_75,code=sm_75",
        "-gencode=arch=compute_80,code=sm_80",  # A100
        "-gencode=arch=compute_86,code=sm_86",
        "-gencode=arch=compute_86,code=sm_89",  # RTX4090
    ]
    sources += sources_cuda
```

## 3. Prepare Datasets

We prepare the nuScenes dataset similar to [bevfusion's instructions](https://github.com/mit-han-lab/bevfusion#data-preparation). Specifically,

1. Download the nuScenes dataset from the [website](https://www.nuscenes.org/nuscenes) and put them in `./data/`. You should have these files:
    ```bash
    data/nuscenes
    ├── maps
    ├── mini
    ├── samples
    ├── sweeps
    ├── v1.0-mini
    └── v1.0-trainval
    ```

> [!TIP]
> You can download the `.pkl` files from [OneDrive](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155157018_link_cuhk_edu_hk/EYF9ZkMHwVZKjrU5CUUPbfYBhC1iZMMnhE2uI2q5iCuv9w?e=QgEmcH). They should be enough for training and testing.

2. Generate mmdet3d annotation files by:

    ```bash
    python tools/create_data.py nuscenes --root-path ./data/nuscenes \
      --out-dir ./data/nuscenes_mmdet3d_2 --extra-tag nuscenes   
    ```
    You should have these files:
    ```bash
    data/nuscenes_mmdet3d_2
    ├── nuscenes_dbinfos_train.pkl (-> ${bevfusion-version}/nuscenes_dbinfos_train.pkl)
    ├── nuscenes_gt_database (-> ${bevfusion-version}/nuscenes_gt_database)
    ├── nuscenes_infos_train.pkl
    └── nuscenes_infos_val.pkl
    ```
    Note: As shown above, some files can be soft-linked with the original version from bevfusion. If some of the files is located in `data/nuscenes`, you can move them to `data/nuscenes_mmdet3d_2` manually.


3. (Optional) To accelerate data loading, we prepared cache files in h5 format for BEV maps. They can be generated through `tools/prepare_map_aux.py` with different configs in `configs/dataset`. For example:
    ```bash
    python tools/prepare_map_aux.py +process=train
    python tools/prepare_map_aux.py +process=val
    ```
    You will have files like `./val_tmp.h5` and `./train_tmp.h5`. You have to rename the cache files correctly after generating them. Our default is
    ```bash
    data/nuscenes_map_aux
    ├── train_26x200x200_map_aux_full.h5 (42G)
    └── val_26x200x200_map_aux_full.h5 (9G)
    ```

4. I prefer as follows:
- download and create dataset on the common directory and linked them in each project directory.

```shell
ln -s ~/DATA/NAS/nfsRoot/Train_Results/img2img-turbo/local_cashe/ local_cashe

ln -s ~/DATA/NAS/nfsRoot/Datasets/nuScenes_Datasets/nuScenes/Full_dataset_v1.0/Trainval/maps maps
ln -s ~/DATA/NAS/nfsRoot/Datasets/nuScenes_Datasets/nuScenes/Full_dataset_v1.0/Trainval/samples samples
ln -s ~/DATA/NAS/nfsRoot/Datasets/nuScenes_Datasets/nuScenes/Full_dataset_v1.0/Trainval/sweeps sweeps
ln -s ~/DATA/NAS/nfsRoot/Datasets/nuScenes_Datasets/nuScenes/Full_dataset_v1.0/Trainval/v1.0-trainval v1.0-trainval
ln -s ~/DATA/NAS/nfsRoot/Datasets/nuScenes_Datasets/nuScenes/Full_dataset_v1.0/Trainval/v1.0-mini v1.0-mini
ln -s ~/DATA/NAS/nfsRoot/Datasets/nuScenes_Datasets/nuScenes/Full_dataset_v1.0/Trainval/panoptic panoptic
ln -s ~/DATA/NAS/nfsRoot/Datasets/nuScenes_Datasets/nuScenes/Full_dataset_v1.0/Trainval/lidarseg lidarseg

ln -s ~/DATA/NAS/nfsRoot/Datasets/nuScenes_Datasets/MagicDrive/data/nuscenes_map_aux nuscenes_map_aux
ln -s ~/DATA/NAS/nfsRoot/Datasets/nuScenes_Datasets/MagicDrive/data/nuscenes_mmdet3d_2 nuscenes_mmdet3d_2
ln -s ~/DATA/HDD8TB/Journal/MagicDrive/data/nuscenes/nuscenes_gt_database nuscenes_gt_database

ln -s ~/DATA/NAS/nfsRoot/Train_Results/MagicDrive magicdrive-log
```


## 4. Pretrained Weights
   
Our training is based on [stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5). We assume you put them at `${ROOT}/pretrained/` as follows:

```bash
{ROOT}/pretrained/stable-diffusion-v1-5/
├── text_encoder
├── tokenizer
├── unet
├── vae
└── ...
```

## 5. Train the model

Launch training with (with 2xA100 80GB):
```bash
cd MagicDrive

accelerate launch --config_file ./configs/accelerator/accelerate_config_2gpu.yaml tools/train.py \
  +exp=224x400 runner=2gpus
```
or
```shell
cd MagicDrive
bash scripts/train.sh
```
During training, you can check tensorboard for the log and intermediate results.

Besides, we provide debug config to test your environment and data loading process :
```bash
accelerate launch --config_file ./configs/accelerator/accelerate_config_2gpu.yaml tools/train.py \
  +exp=224x400 runner=debug runner.validation_before_run=true
```
or
```shell
cd MagicDrive
bash scripts/train_debug.sh
```
## 6. Convert Model Files
save pytorch model from accelerate checkpoint files

```shell
accelerate launch --config_file ./configs/accelerator/accelerate_config_1gpu.yaml \
  tools/save_pytorch_model_from_accelerate_checkpoint.py \
  resume_from_checkpoint=./magicdrive-log/SDv1.5mv-rawbox_2024-12-13_21-38_224x400/checkpoint-160000 \
  +exp=224x400 runner=2gpus
```
or
```shell
cd MagicDrive
bash scripts/save_pytorch_model_from_accelerate_checkpoint.sh
```

## 7. Test the model
After training, you can test your model for driving view generation through:
```bash
python tools/test.py resume_from_checkpoint=${YOUR MODEL}
# take our the 224x400 model checkpoint as an example
python tools/test.py resume_from_checkpoint=./pretrained/SDv1.5mv-rawbox_2023-09-07_18-39_224x400
```
or
```shell
python tools/inference_test_hkkim.py resume_from_checkpoint=./magicdrive-log/model_convert/SDv1.5mv-rawbox_2024-12-17_23-16_224x400
```
or
```shell
cd MagicDrive
bash scripts/inference_test_hkkim.sh
```