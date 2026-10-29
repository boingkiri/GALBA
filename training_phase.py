
import os

import torch
import torch.nn as nn

import hydra
from omegaconf import DictConfig, OmegaConf

from models.target_model_framework import TargetModelFramework
# from utils.data_utils import DataContainer, DataModule
from utils.data_utils import DataModule
# from utils.trainer_utils import get_trainer, get_previous_cycle_dir, \
from utils.trainer_utils import get_trainer, \
                                get_best_metrics_from_trainer, get_model_checkpoint_callback, \
                                get_best_ckpt_from_last_cycle
from utils.cycle_status_utils import CycleStatusUtils
from generative_oracle.generative_oracle_container import GenerativeOracleContainer

import lightning as L

import wandb
import datetime

from utils.wandb_logger_utils import create_logger, get_group_name, get_project_name
import gc

from accelerate import Accelerator


# --- For TPU-specific functionality ---
try:
    import torch_xla.core.xla_model as xm
# Check for manual disable or if explicitly on TPU
    # TMP
    os.environ["WANDB_START_METHOD"] = "thread"
    wandb.setup()
    _USE_XLA = True
    # import gcsfs
    from google.cloud import storage
    # storage_client = storage.Client()
    storage_client = storage.Client()
    GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "generative_active_learning")
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
except ImportError:
    # This allows the code to run in a non-TPU environment without crashing
    xm = None
    _USE_XLA = False

def free_memory(model_to_delete, accelerator: Accelerator):
    """
    Deletes a model and clears the appropriate memory cache for GPU or TPU.

    Args:
        model_to_delete: The model object to delete.
        accelerator: The Accelerator object which knows the device type.
    """
    device_type = accelerator.device.type
    print(f"Attempting to free memory for device type: {device_type}")

    # Step 1: Delete the model and any other tensors from CPU/GPU/TPU memory
    del model_to_delete
    
    # Step 2: Force Python's garbage collector to clean up deleted objects
    gc.collect()

    # Step 3: Call the device-specific cache clearing function
    if device_type == 'cuda':
        # For NVIDIA GPUs, clear the CUDA cache
        print("Clearing CUDA memory cache...")
        torch.cuda.empty_cache()
        print("CUDA memory cache cleared.")
    elif device_type == 'xla':
        # For TPUs, mark the step to trigger XLA's memory reclamation
        if xm:
            print("Marking XLA step to free TPU memory...")
            xm.mark_step()
            print("XLA step marked.")
        else:
            print("torch_xla not found. Cannot free TPU memory.")
    else:
        print(f"No specific memory freeing action for device type: {device_type}")


def collect_all_generated_data_in_current_dir(data_module, current_dir, current_idx):
    if not _USE_XLA:
        for cycle_idx in os.listdir(current_dir):
            # if "logging" in cycle_idx:
            #     continue
            if "cycle" not in cycle_idx \
                and "initial_generated_data" not in cycle_idx:
                continue
            cycle_dir = os.path.join(current_dir, cycle_idx)
            if cycle_idx == "initial_generated_data":
                data_module.add_generated_dataset(os.path.join(cycle_dir, "generated_dataset"))
                continue

            walking_idx = int(cycle_idx.replace("cycle_", ""))
            if walking_idx >= current_idx:
                continue
            if os.path.isdir(cycle_dir):
                data_module.add_generated_dataset(os.path.join(cycle_dir, "generated_dataset"))
    else:
        tar_list = []
        if "imagenette" in current_dir:
            for blob in bucket.list_blobs(prefix="data/imagenette/initial_generated_data"):
                blob_name = blob.name
                if blob_name.endswith(".tar"):
                    # tar_list.append(os.path.join("gs://generative_active_learning", blob_name))
                    tar_list.append(os.path.join(f"gs://{GCS_BUCKET_NAME}", blob_name))
        elif "imagenet_100" in current_dir:
            for blob in bucket.list_blobs(prefix="data/imagenet_100/initial_generated_data"):
                blob_name = blob.name
                if blob_name.endswith(".tar"):
                    # tar_list.append(os.path.jo/in("gs://generative_active_learning", blob_name))
                    tar_list.append(os.path.join(f"gs://{GCS_BUCKET_NAME}", blob_name))

            # for blob in bucket.list_blobs(prefix="data/imagenet_100/initial_generated_data/images"):
            #     blob_name = blob.name
            #     if blob_name.endswith(".tar"):
            #         tar_list.append(os.path.join("gs://generative_active_learning", blob_name))
        else:
            NotImplementedError("TPU generated data collection not implemented for this dataset yet.")

        # print("current_dir", current_dir)
        print("Tar list for generated dataset (intermediate):", tar_list)
        print("length of tar for generated dataset:", len(tar_list))
        for blob in bucket.list_blobs(prefix=current_dir):
            blob_name = blob.name
            if blob_name.endswith(".tar"):
                # tar_list.append(os.path.join("gs://generative_active_learning", blob_name))
                tar_list.append(os.path.join(f"gs://{GCS_BUCKET_NAME}", blob_name))
        print("Tar list for generated dataset:", tar_list)
        print("length of tar for generated dataset:", len(tar_list))
        data_module.add_generated_dataset_from_tar_urls(tar_list)

    # return os.path.join(current_dir, f"cycle_{last_cycle}", "best_model.ckpt")


@hydra.main(version_base=None, config_path="configs")
def train(cfg: DictConfig) -> None:
    start_datetime = datetime.datetime.now().strftime('%y%m%d_%H%M%S')

    # 0. Set random seeds for reproducibility
    # seeds = OmegaConf.to_container(cfg.seed, resolve=True)
    # if type(seeds) is int:
    #     seeds = [seeds]
    # elif type(seeds) is list:
    #     seeds = seeds
    # else:
    #     raise ValueError(f"Unsupported seed type: {type(seeds)}. Expected int or list of ints.")
    seed = cfg.get("seed", 42)
    seed = int(seed)

    # 1. Load the dataset
    # data_module = DataModule(cfg, num_processes=accelerator.num_processes)
    
    # 2. Initialize the model
    last_best_ckpt_path = None
    workspace_name = f"{start_datetime}_" + get_group_name(cfg) + f"/seed_{seed}"
    default_root_dir = cfg.get("default_root_dir", os.path.join(get_project_name(cfg), workspace_name))

    if not os.path.exists(default_root_dir):
        os.makedirs(default_root_dir, exist_ok=True)

    last_best_ckpt_path, last_cycle, is_done = get_best_ckpt_from_last_cycle(default_root_dir, from_gcs=_USE_XLA)

    current_cycle = last_cycle + 1 if last_cycle is not None else 0
    print("Current cycle:", current_cycle)

    current_cycle_dir = os.path.join(f"{default_root_dir}/cycle_{current_cycle}")
    if not os.path.exists(current_cycle_dir):
        os.makedirs(current_cycle_dir, exist_ok=True)

    # CycleStatusUtils.create_cycle_status(os.path.join(f"{default_root_dir}/cycle_{current_cycle}"))
    CycleStatusUtils.load_cycle_status(os.path.join(f"{default_root_dir}/cycle_{current_cycle}"))

    logger = create_logger(cfg, seed, default_root_dir)
    trainer, _, logger = get_trainer(
            cfg, cycle=current_cycle, logger=logger, 
            start_datetime=start_datetime, seed=seed,
            default_root_dir=default_root_dir, num_devices=cfg.num_devices, 
            num_nodes=cfg.get("num_nodes", 1), is_done=is_done)
    
    num_nodes = cfg.get("num_nodes", 1)
    # data_module = DataModule(cfg, num_processes=cfg.num_devices, seed=seed)
    data_module = DataModule(cfg, num_processes=cfg.num_devices * num_nodes, seed=seed)
    # collect_all_generated_data_in_current_dir(data_module, default_root_dir)
    collect_all_generated_data_in_current_dir(data_module, default_root_dir, current_idx=current_cycle)


    L.pytorch.seed_everything(seed, workers=True)
    model_container = TargetModelFramework(cfg, cycle=current_cycle)
    if cfg.get("load_best_checkpoint", False) and last_best_ckpt_path:
        model_container = TargetModelFramework.load_from_checkpoint(last_best_ckpt_path, strict=True)
        model_container.set_cycle(cycle=current_cycle)

    # Target model training
    print("Training the target model...")
    # print("Dataloader length:", data_module.train_dataloader_len)

    # For imagenet-100
    # if "imagenet" in cfg.dataset.name.lower() or "imagenette" in cfg.dataset.name.lower():
    #     trainer.fit(
    #         model=model_container,
    #         datamodule=data_module,
    #         ckpt_path=last_best_ckpt_path
    #     )
    # else:
    #     trainer.fit(
    #         model=model_container,
    #         datamodule=data_module,
    #     )
    trainer.fit(
        model=model_container,
        datamodule=data_module,
        ckpt_path=last_best_ckpt_path
    )

    # Extract validation accuracy
    # log = {
    #     "num_labeled_samples": len(data_module.train_dataset),
    #     "cycle": current_cycle,
    #     "step": current_cycle,
    # }
    log = {
        "cycle": current_cycle,
        "step": current_cycle,
    }


    try:
        if trainer.global_rank == 0:
            mc = trainer.checkpoint_callback
            
            # [수정 1] best_model_path는 보통 잘 동기화되므로 우선 확인
            best_path = mc.best_model_path if mc else None
            
            if not best_path:
                trainer.print("[WARN] best_model_path is empty. Checkpoint might not have been saved.")
                score_val = None
            else:
                # [수정 2] 메모리 변수(best_model_score) 대신 저장된 파일에서 정보를 로드
                # TPU 환경에서는 trainer 객체의 상태가 업데이트 안 됐을 확률이 높음
                score = mc.best_model_score
                monitor_key = mc.monitor
                
                if score is None:
                    trainer.print(f"[INFO] best_model_score is None on Main Process (TPU issue). Loading from ckpt: {best_path}")
                    try:
                        # CPU로 맵핑하여 가볍게 로드
                        ckpt = torch.load(best_path, map_location="cpu", weights_only=False) 
                        
                        # PyTorch Lightning 체크포인트 내부 구조에서 score 추출
                        # 보통 'callbacks' 키 안에 ModelCheckpoint의 상태가 저장됨
                        # 혹은 간단히 monitor_key에 해당하는 값이 loop output에 있을 수 있으나,
                        # 가장 확실한 건 ModelCheckpoint가 저장한 score를 찾는 것입니다.
                        
                        # ModelCheckpoint의 state_key를 찾아야 하는데, 
                        # 보통 monitor_key가 체크포인트의 'callbacks' -> 'ModelCheckpoint{...}' -> 'best_model_score' 에 있음
                        # 하지만 버전마다 다르므로, 가장 안전한 건 ckpt['callbacks']를 순회하거나
                        # 단순히 파일이 존재한다는 것은 그게 Best라는 뜻이므로, 
                        # 당시의 metric 값을 모른다면 로그를 믿어야 할 수도 있습니다.
                        
                        # 대안: 만약 score 값을 꼭 숫자로 얻어야 한다면:
                        found_score = False
                        if "callbacks" in ckpt:
                            for key, val in ckpt["callbacks"].items():
                                if "ModelCheckpoint" in key:
                                    if "best_model_score" in val:
                                        score_val = val["best_model_score"]
                                        found_score = True
                                        break
                        
                        if not found_score:
                             # 만약 callbacks dict에서 못 찾으면, 그냥 None 처리하거나 로그 확인 유도
                             score_val = 0.0 # 혹은 적절한 default
                             trainer.print("[WARN] Could not retrieve score from checkpoint file.")
                        else:
                             # Tensor인 경우 float 변환
                             if torch.is_tensor(score_val):
                                 score_val = score_val.item()
                                 
                    except Exception as e:
                        trainer.print(f"[ERR] Failed to load checkpoint to retrieve score: {e}")
                        score_val = None
                else:
                    # GPU 등에서 정상적으로 가져와진 경우
                    try:
                        score_val = score.item()
                    except AttributeError:
                        score_val = float(score)

                # 로그 업데이트 및 출력 (score_val이 유효할 때만)
                if score_val is not None:
                    best_metrics = {f"best_{monitor_key}": score_val}
                    log.update(best_metrics)
                    
                    logger.experiment.log(log)
                    trainer.print("-------- Cycle Summary -------")
                    trainer.print(f"Cycle {current_cycle} completed. Best metrics: {best_metrics}")
                    trainer.print(f"Best Checkpoint Path: {best_path}")
                    trainer.print("-----------------------------")

        if xm is not None:
            xm.rendezvous("end_of_train_logging")
            
    finally:
        if trainer.is_global_zero:
            wandb.finish()

            print("default_root_dir: ",default_root_dir)
            CycleStatusUtils.save_cycle_status(
                cycle_dir=os.path.join(f"{default_root_dir}/cycle_{current_cycle}"),
                training_phase_done=True
            )

if __name__ == "__main__":
    train()