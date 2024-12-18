import os
import sys
import logging
from dotenv import load_dotenv
load_dotenv('/home/hyunkoo/DATA/ssd8tb/Journal/MagicDrive/.env')

print(os.environ['HF_HOME'])

import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch

from mmdet3d.datasets import build_dataset
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed

sys.path.append(".")  # 필요하다면 적절히 수정
import magicdrive.dataset.pipeline
from magicdrive.misc.common import load_module


def set_logger(global_rank, logdir):
    if global_rank == 0:
        return
    logging.info(f"reset logger for {global_rank}")
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
    file_path = os.path.join(logdir, f"train.{global_rank}.log")
    handler = logging.FileHandler(file_path)
    handler.setFormatter(formatter)
    root.addHandler(handler)


@hydra.main(version_base=None, config_path="../configs", config_name="model_convert_hkkim_config")
def main(cfg: DictConfig):
    # 기존과 동일한 환경 설정
    logging.getLogger().setLevel(logging.DEBUG)
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler) or cfg.try_run:
            handler.setLevel(logging.DEBUG)
        else:
            handler.setLevel(logging.INFO)
    logging.getLogger("shapely.geos").setLevel(logging.WARN)
    logging.getLogger("asyncio").setLevel(logging.INFO)
    logging.getLogger("accelerate.tracking").setLevel(logging.INFO)
    logging.getLogger("numba").setLevel(logging.WARN)
    logging.getLogger("PIL").setLevel(logging.WARN)
    logging.getLogger("matplotlib").setLevel(logging.WARN)

    setattr(cfg, "log_root", HydraConfig.get().runtime.output_dir)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.accelerator.gradient_accumulation_steps,
        mixed_precision=cfg.accelerator.mixed_precision,
        log_with=cfg.accelerator.report_to,
        project_dir=cfg.log_root,
        kwargs_handlers=[ddp_kwargs],
    )
    set_logger(accelerator.process_index, cfg.log_root)
    set_seed(cfg.seed)

    # dataset 필요 없다면 생략 가능 (단, runner 초기화에 필요하다면 남겨둬야 함)
    train_dataset = build_dataset(
        OmegaConf.to_container(cfg.dataset.data.train, resolve=True)
    )
    val_dataset = build_dataset(
        OmegaConf.to_container(cfg.dataset.data.val, resolve=True)
    )

    # runner 초기화
    runner_cls = load_module(cfg.model.runner_module)
    runner = runner_cls(cfg, accelerator, train_dataset, val_dataset)
    runner.set_optimizer_scheduler()
    runner.prepare_device()

    # 여기서 이미 학습 완료된 체크포인트를 로드
    # cfg.resume_from_checkpoint 를 통해 체크포인트 경로를 받아온다고 가정
    if not cfg.resume_from_checkpoint:
        raise ValueError("resume_from_checkpoint 경로를 지정해주세요.")
    load_path = cfg.resume_from_checkpoint
    accelerator.load_state(load_path)

    # unwrap_model 로 모델 추출
    controlnet = accelerator.unwrap_model(runner.controlnet)

    # 모델 저장 (controlnet_dir은 original code에서 cfg.model.controlnet_dir 로 지정)
    save_dir = os.path.join(cfg.log_root, "controlnet")
    os.makedirs(save_dir, exist_ok=True)
    controlnet.save_pretrained(save_dir)
    logging.info(f"Model saved to: {save_dir}")

    # unwrap_model 로 모델 추출
    unet = accelerator.unwrap_model(runner.unet)

    # 모델 저장 (controlnet_dir은 original code에서 cfg.model.controlnet_dir 로 지정)
    save_dir = os.path.join(cfg.log_root, "unet")
    os.makedirs(save_dir, exist_ok=True)
    unet.save_pretrained(save_dir)
    logging.info(f"Model saved to: {save_dir}")


if __name__ == "__main__":
    main()
