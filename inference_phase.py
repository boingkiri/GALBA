
import os
import gc
import json
import random
import shutil
import datetime
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

import torch
import lightning as L

import accelerate
from accelerate import Accelerator

# from utils.data_utils import DataModule
from utils.data_utils import DataModuleDeterministic, DataModuleDeterministicGeneratedDataForHoldout, \
                            DataModuleDeterministicGeneratedDataForHoldoutBalanced, \
                            DataModuleDeterministicClasswiseHoldout, DataModuleDeterministicClasswiseHoldoutUsingTestDataset, \
                            DataModuleDeterministicInitialTrainingAsHoldout, DataModuleDeterministicClasswiseGeneratedDataForHoldoutBalanced, \
                            DataModuleDeterministicClasswiseGeneratedDataSelectedByCLIPScore, \
                            DataModuleDeterministicGeneratedDataSelectedByCLIPScore
from utils.trainer_utils import get_best_ckpt_from_last_cycle
from utils.wandb_logger_utils import create_wandb_logger, get_group_name, get_project_name, create_logger
from utils.cycle_status_utils import CycleStatusUtils

from generative_oracle.generative_oracle_container import GenerativeOracleContainer

import wandb
import sys
import time

# --- For TPU-specific functionality ---
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    _USE_XLA = True
    from google.cloud import storage
    # storage_client = storage.Client(project=os.environ.get("GCS_PROJECT_NAME", "generative-active-learning"))
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
    print(f"Attempting to free memory for device type: {device_type}", file=sys.stderr)

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

    # return os.path.join(current_dir, f"cycle_{last_cycle}", "best_model.ckpt")

def collect_all_previous_generated_data_in_current_dir(data_module, current_dir, current_cycle):
    current_cycle = f"cycle_{current_cycle}"
    if _USE_XLA:
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
                    # tar_list.append(os.path.join("gs://generative_active_learning", blob_name))
                    tar_list.append(os.path.join(f"gs://{GCS_BUCKET_NAME}", blob_name))
        else:
            NotImplementedError("TPU generated data collection not implemented for this dataset yet.")

        # print("current_dir", current_dir)
        for blob in bucket.list_blobs(prefix=current_dir):
            blob_name = blob.name
            if blob_name.endswith(".tar") and current_cycle not in blob_name:
                # tar_list.append(os.path.join("gs://generative_active_learning", blob_name))
                tar_list.append(os.path.join(f"gs://{GCS_BUCKET_NAME}", blob_name))
        print("Tar list for generated dataset:", tar_list, file=sys.stderr)
        print("length of tar for generated dataset:", len(tar_list), file=sys.stderr)
        data_module.add_generated_dataset_from_tar_urls(tar_list)
    else:
        for cycle_idx in os.listdir(current_dir):
            if "logging" in cycle_idx:
                continue
            if current_cycle == cycle_idx:
                continue
            cycle_dir = os.path.join(current_dir, cycle_idx)
            if os.path.isdir(cycle_dir):
                data_module.add_generated_dataset(os.path.join(cycle_dir, "generated_dataset"))

def broadcast_to_all(data, accelerator: Accelerator):
    """
    Broadcasts data to all processes.

    Args:
        data: The data to broadcast.
        accelerator: The Accelerator object which knows the device type.
    """
    if accelerator.num_processes > 1:
        if accelerator.is_main_process:
            object_to_broadcast = [data]
        else:
            object_to_broadcast = [None]
        accelerate.utils.broadcast_object_list(object_to_broadcast, from_process=0)
        data = object_to_broadcast[0]
    return data



@hydra.main(version_base=None, config_path="configs")
def inference(cfg: DictConfig) -> None:
    accelerator = Accelerator()
    start_datetime = datetime.datetime.now().strftime('%y%m%d_%H%M%S')

    if _USE_XLA:
        os.environ["RANK"] = str(xr.global_ordinal())
        os.environ["WORLD_SIZE"] = str(xr.world_size())
    
    # 0. Set random seeds for reproducibility
    
    seed = cfg.get("seed", 42)
    seed = int(seed)
        
    last_best_ckpt_path = None

    # 2. Initialize the model
    # wandb_logger = create_wandb_logger(cfg, seed)
    workspace_name = f"{start_datetime}_" + get_group_name(cfg) + f"/seed_{seed}"

    # TMP: TODO -> need to make another function
    # default_root_dir = cfg.get("default_root_dir", os.path.join(get_project_name(cfg), workspace_name))
    if cfg.get("default_root_dir", "") == "":
        cfg.default_root_dir = os.path.join(get_project_name(cfg), workspace_name)

    # Always ensure the run directory exists — an explicit default_root_dir may
    # point at a fresh path (e.g. a new seed) whose parents don't exist yet.
    if accelerator.is_main_process:
        os.makedirs(os.path.join(cfg.default_root_dir), exist_ok=True)

    accelerator.wait_for_everyone()
    cfg.default_root_dir = broadcast_to_all(cfg.default_root_dir, accelerator)
    default_root_dir = cfg.default_root_dir
    default_root_dir = broadcast_to_all(default_root_dir, accelerator)
    print(f"Default root directory for logs and checkpoints: {default_root_dir}", file=sys.stderr)

    last_best_ckpt_path, last_cycle, is_done = get_best_ckpt_from_last_cycle(default_root_dir, from_gcs=_USE_XLA)
    
    assert is_done is not False

    print("Last best checkpoint path: ", last_best_ckpt_path, file=sys.stderr)


    # logger = create_logger(cfg, seed, default_root_dir)
    logger = create_logger(cfg, seed, default_root_dir, accelerator)

    # 3. Train the model
    cycle_idx = last_cycle + 1 if last_cycle is not None else -1
    # accumulated_image_count = 0 if cycle_idx == -1 \
    #     else cfg.num_initial_labeled_generated_samples + cfg.generative_oracle.num_generated_sample_per_cycle * (cycle_idx - 1)
    accumulated_image_count = 0 if last_cycle is None \
        else cfg.num_initial_labeled_generated_samples + cfg.generative_oracle.num_generated_sample_per_cycle * (cycle_idx - 1)
    print("accumulated_image_count: ", accumulated_image_count, file=sys.stderr)

    if cfg.dataset.get("generated_data_for_holdout", False) and cycle_idx > 0:
        if cfg.dataset.get("selected_by_clip_score", False) and cfg.dataset.get("classwise_holdout", False):
            data_module = DataModuleDeterministicClasswiseGeneratedDataSelectedByCLIPScore(
                cfg, num_processes=accelerator.num_processes, seed=seed, cycle=cycle_idx-1, accelerator=accelerator)
        elif cfg.dataset.get("selected_by_clip_score", False):
            data_module = DataModuleDeterministicGeneratedDataSelectedByCLIPScore(
                cfg, num_processes=accelerator.num_processes, seed=seed, accelerator=accelerator)
        elif cfg.dataset.get("classwise_holdout", False) and cfg.dataset.get("balanced_holdout", False):
            data_module = DataModuleDeterministicClasswiseGeneratedDataForHoldoutBalanced(cfg, num_processes=accelerator.num_processes, seed=seed, cycle=cycle_idx-1)
        elif cfg.dataset.get("balanced_holdout", False):
            data_module = DataModuleDeterministicGeneratedDataForHoldoutBalanced(cfg, num_processes=accelerator.num_processes, seed=seed)
        else:
            data_module = DataModuleDeterministicGeneratedDataForHoldout(cfg, num_processes=accelerator.num_processes, seed=seed)
    elif cfg.dataset.get("classwise_holdout", False):
        data_module = DataModuleDeterministicClasswiseHoldout(cfg, num_processes=accelerator.num_processes, seed=seed)
    elif cfg.dataset.get("classwise_holdout_using_test_dataset", False):
        data_module = DataModuleDeterministicClasswiseHoldoutUsingTestDataset(cfg, num_processes=accelerator.num_processes, seed=seed)
    elif cfg.dataset.get("classwise_holdout_using_initial_real_data", False):
        data_module = DataModuleDeterministicInitialTrainingAsHoldout(cfg, num_processes=accelerator.num_processes, seed=seed)
    else:
        data_module = DataModuleDeterministic(cfg, num_processes=accelerator.num_processes, seed=seed)

    data_module.prepare_data()
    data_module.setup(stage="fit")
    data_module.setup(stage="test")
    data_module.setup(stage="predict")
    collect_all_previous_generated_data_in_current_dir(data_module, default_root_dir, cycle_idx-1)

    # if cfg.dataset.get("generated_data_for_holdout", False) and cycle_idx > 0:
    #     data_module.set_holdout_dataset_from_train_dataset()

    ## TMP
    data_module.batch_size = 1

    generating_seed = seed * 10000 + cycle_idx # TODO: nasty code. need to be changed
    generating_seed = int(str(accelerator.process_index) + str(generating_seed))
    print(f"Generating seed: {generating_seed}", file=sys.stderr)
    L.pytorch.seed_everything(generating_seed, workers=True)
        

    # 3.1. Generate new data if cycle_idx > 0

    # if cycle_idx > 0 or cfg.num_initial_labeled_generated_samples > 0:
    # if cycle_idx == -1: # TMP
    #     # print(f"[Rank {accelerator.process_index}] Inference phase completed.", file=sys.stderr)
    #     # accelerator.wait_for_everyone()
    #     # print("All processes have completed inference phase.", file=sys.stderr)
    #     pass # TMP
    # elif cycle_idx > 0 or cfg.num_initial_labeled_generated_samples > 0:
    if cycle_idx > 0 or (cycle_idx == -1 and cfg.num_initial_labeled_generated_samples > 0):
    # if cycle_idx > 0 or cfg.num_initial_labeled_generated_samples > 0:
        print("Generating new data...")

        # At cycle -1 no target model has been trained yet, so any guidance
        # (gradient-entropy) or FKS particle resampling (BALD) would be
        # wasted compute or produce zeros. Force the fast no-guidance pipeline
        # so all three variants share an identical initial-generation cost.
        if cycle_idx == -1:
            with open_dict(cfg):
                cfg.guidance_model.guidance_model_type = "no_guidance"
                cfg.generative_oracle.guidance_type = "gradient"
                cfg.generative_oracle.fkd_args = None
            print("[cycle -1] Overriding to no_guidance/gradient pipeline for "
                  "initial data generation.", file=sys.stderr)

        guidance_model_kwargs = {}
        guidance_model_kwargs = {
            "datamodule": data_module,
            "model_ckpt_path": last_best_ckpt_path,
            "dummy_flag": True if last_cycle is None else False
        }
        generative_oracle_container = GenerativeOracleContainer(
            cfg=cfg,
            accelerator=accelerator,
            cycle_idx=cycle_idx,
            logger=logger,
            guidance_model_ckpt_path=last_best_ckpt_path,
            accumulated_image_count=accumulated_image_count,
            **guidance_model_kwargs
        )

        # Cycle -1 seeds the run: dump into `initial_generated_data/` and use
        # `num_initial_labeled_generated_samples`, matching the folder layout
        # training_phase.py expects. Otherwise use `cycle_{N-1}/`.
        if cycle_idx == -1:
            previous_cycle_dir = os.path.join(default_root_dir, "initial_generated_data")
            num_samples_to_generate = cfg.num_initial_labeled_generated_samples
        else:
            previous_cycle_dir = os.path.join(default_root_dir, f"cycle_{cycle_idx - 1}")
            num_samples_to_generate = cfg.generative_oracle.num_generated_sample_per_cycle
        print("Generating new data for the cycle...")

        generation_start_time = time.time()
        accumulated_image_count = generative_oracle_container.generate_samples(
            previous_cycle_dir,
            num_samples_to_generate,
            # Force freedom_guidance_scale=0 at cycle -1 so the initial set is
            # generated without gradient guidance.
            guidance_scale=0.0 if cycle_idx == -1 else None,
        )

        generation_end_time = time.time()
        generation_duration = generation_end_time - generation_start_time
        print(f"Data generation for cycle {cycle_idx} took {generation_duration:.2f} seconds.", file=sys.stderr)
        logger.experiment.log({"data_generation_time": generation_duration, "step": cycle_idx-1})

        if accelerator.is_main_process:
            CycleStatusUtils.save_cycle_status(
                cycle_dir=previous_cycle_dir,
                inference_phase_done=True
            )

        else:
            print("TPU environment detected. Please ensure that the initial generated data is prepared in the specified directory.", file=sys.stderr)


    print(f"[Rank {accelerator.process_index}] Inference phase completed.", file=sys.stderr)
    accelerator.wait_for_everyone()
    print("All processes have completed inference phase.", file=sys.stderr)
    # if accelerator.is_main_process:
    if accelerator.is_local_main_process:
        # print(f"Total number of generated samples used for training: {accumulated_image_count}")
        print("default_root_dir: ",default_root_dir)

    return default_root_dir
  
if __name__ == "__main__":
    default_root_dir = inference()
    # print(default_root_dir)