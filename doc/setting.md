```shell
conda create -n magicdrive python==3.10.15 -y
conda activate magicdrive
```
pip install -r requirements/py10cu12_1/requirements.txt 
git lfs install
git clone https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5

cd third_party/diffusers
pip install .

cd third_party/
git clone https://github.com/mit-han-lab/bevfusion && cd bevfusion

Now, you should be able to run our demo.

### Q3: [Error] nvcc fatal   : Unsupported gpu architecture 'compute_80'

- This may appear when you install bevfusion (mmdet3d) on cuda10.2. The latest version of bevfusion supports Ampere GPUs by hard-coding compile parameters, leading to error when compiled with cuda10.2. One can get rid of this error by comment these lines in `third_party/bevfusion/setup.py (L19)`.
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

python setup.py develop



```shell
pip uninstall torch torchvision tensorboard hydra-core urllib3 h5py einops accelerate transformers mmcv-full openmim Pillow nuscenes-devkit numba llvmlite numpy  llvmlite mmdet opencv-python ninja diffusers
```

==========================

conda create -n magicdrive37 python==3.7 -y
=======================================================================

conda create -n mdrive39 python==3.9 -y

data
```shell
ln -s ~/DATA/NAS/nfsRoot/Train_Results/img2img-turbo/local_cashe/ local_cashe

ln -s ~/DATA/NAS/nfsRoot/Datasets/nuScenes_Datasets/nuScenes/Full_dataset_v1.0/Trainval/maps maps
ln -s ~/DATA/NAS/nfsRoot/Datasets/nuScenes_Datasets/nuScenes/Full_dataset_v1.0/Trainval/samples samples
ln -s ~/DATA/NAS/nfsRoot/Datasets/nuScenes_Datasets/nuScenes/Full_dataset_v1.0/Trainval/sweeps sweeps
ln -s ~/DATA/NAS/nfsRoot/Datasets/nuScenes_Datasets/nuScenes/Full_dataset_v1.0/Trainval/v1.0-trainval v1.0-trainval
ln -s ~/DATA/NAS/nfsRoot/Datasets/nuScenes_Datasets/nuScenes/Full_dataset_v1.0/Trainval/v1.0-mini v1.0-mini
ln -s ~/DATA/NAS/nfsRoot/Datasets/nuScenes_Datasets/nuScenes/Full_dataset_v1.0/Trainval/panoptic panoptic
ln -s ~/DATA/NAS/nfsRoot/Datasets/nuScenes_Datasets/nuScenes/Full_dataset_v1.0/Trainval/lidarseg lidarseg
```