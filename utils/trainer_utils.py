import os
import sys
import datetime

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from .wandb_logger_utils import create_wandb_logger
# from .callbacks_utils import ResetEarlyStoppingState, ModifyModelCheckpointFormatState, \
#     CustomMaxStepsCallback, GCSUploaderCallback, EarlyStoppingRenameCallback
from .callbacks_utils import ResetEarlyStoppingState, ModifyModelCheckpointFormatState, \
    CustomMaxStepsCallback, GCSUploaderCallback

from .cycle_status_utils import CycleStatusUtils
# from lightning.pytorch.plugins.io import XLACheckpointIO

# try:
#     import torch_xla
#     _HAS_XLA = True
# except Exception:
#     _HAS_XLA = False


# You should also fix the check for XLA
try:
    import torch_xla
    from google.cloud import storage
    _USE_XLA = True
    # Create the filesystem object if XLA is available
    storage_client = storage.Client()
    bucket_name=os.environ.get("GCS_BUCKET_NAME", "generative_active_learning")
    bucket = storage_client.bucket(bucket_name)
except Exception:
    _USE_XLA = False
    gcs_filesystem = None # Not needed for local


def get_best_ckpt_from_dir(current_dir, cycle_idx):
    cycle_dir = os.path.join(current_dir, f"cycle_{cycle_idx}")
    for cycle_file_elem in os.listdir(cycle_dir):
        if "epoch" in cycle_file_elem and cycle_file_elem.endswith(".ckpt"):
            return os.path.join(cycle_dir, cycle_file_elem)
    raise ValueError(f"The best ckpt does not exist in cycle {cycle_idx}.")

def is_cycle_finalized(current_dir, cycle_idx):
    cycle_dir = os.path.join(current_dir, f"cycle_{cycle_idx}")
    if not os.path.exists(cycle_dir):
        return False
    for cycle_file_elem in os.listdir(cycle_dir):
        if "ckpt" in cycle_file_elem and "final" in cycle_file_elem: # TODO: done 추가해야함
            return True
    return False

def is_cycle_reach_patience(current_dir, cycle_idx):
    # if the cycle reaches patience, then it creates "generated_dataset" directory.
    cycle_dir = os.path.join(current_dir, f"cycle_{cycle_idx}")
    generated_data_dir = os.path.join(cycle_dir, "generated_dataset")
    return os.path.exists(generated_data_dir)

def get_best_ckpt_from_last_cycle(current_dir, from_gcs=False): # best_checkpoint_path, last_cycle, is_done
    last_cycle = -1

    if not from_gcs:
        for cycle_idx in os.listdir(current_dir):
            if "cycle" not in cycle_idx:
                continue
            cycle_idx = cycle_idx.split("_")[-1]
            if cycle_idx.isdigit():
                last_cycle = max(last_cycle, int(cycle_idx))
        # return last_cycle
        if last_cycle == -1:
            return None, None, None

        for cycle_idx in range(last_cycle, -1, -1):
            cycle_dir = os.path.join(current_dir, f"cycle_{cycle_idx}")
            is_training_state_done = CycleStatusUtils.is_training_phase_done(cycle_dir)
            if is_training_state_done:
                # return get_best_ckpt_from_dir(current_dir, cycle_idx), last_cycle, True
                return get_best_ckpt_from_dir(current_dir, cycle_idx), cycle_idx, True

        # ValueError("The best ckpt does not exist in the last cycle.")
        print("The best ckpt does not exist in the last cycle.")
        return None, None, None

    else:
        blobs = bucket.list_blobs(prefix=f"{current_dir}/")
        for blob in blobs:
            # blobs contains all of the files in the current_dir, including those in subdirectories
            print(blob.name, file=sys.stderr)
            path_parts = blob.name.split('/')
            for part in path_parts:
                if "cycle_" in part:
                    cycle_idx = part.split("_")[-1]
                    if cycle_idx.isdigit():
                        last_cycle = max(last_cycle, int(cycle_idx))
        if last_cycle == -1:
            return None, None, None

        for cycle_idx in range(last_cycle, -1, -1):
            cycle_dir = f"{current_dir}/cycle_{cycle_idx}"
            is_training_state_done = CycleStatusUtils.is_training_phase_done(cycle_dir)
            blobs = bucket.list_blobs(prefix=f"{cycle_dir}/")
            training_reached = any("epoch" in blob.name and blob.name.endswith(".ckpt") for blob in blobs)
            if is_training_state_done:
                # return get_best_ckpt_from_dir(current_dir, cycle_idx), last_cycle, True
                blobs = bucket.list_blobs(prefix=f"{cycle_dir}/")
                for blob in blobs:
                    if "epoch" in blob.name and blob.name.endswith(".ckpt"):
                        return f"gs://{bucket_name}/{blob.name}", cycle_idx, True
            elif training_reached:
                # the training state is not done yet.
                blobs = bucket.list_blobs(prefix=f"{cycle_dir}/")
                for blob in blobs:
                    if "epoch" in blob.name and blob.name.endswith(".ckpt"):
                        return f"gs://{bucket_name}/{blob.name}", cycle_idx, False
        if last_cycle == 0:
            print(f"Very first cycle training.", file=sys.stderr)
            return None, None, False
        
        ValueError("The best ckpt does not exist in the last cycle.")

def get_best_ckpt_from_early_non_final_phase_cycle(current_dir):
    last_cycle = -1

    if not _USE_XLA:
        for cycle_idx in os.listdir(current_dir):
            if "cycle" not in cycle_idx:
                continue
            cycle_idx = cycle_idx.split("_")[-1]
            if cycle_idx.isdigit():
                last_cycle = max(last_cycle, int(cycle_idx))

        # for cycle_idx in range(last_cycle, -1, -1):
        non_final_phase_cycle = None
        for cycle_idx in range(0, last_cycle + 1):
        # for cycle_idx in range(last_cycle, -1, -1):
            cycle_dir = os.path.join(current_dir, f"cycle_{cycle_idx}")
            is_training_phase_done = CycleStatusUtils.is_training_phase_done(cycle_dir)
            is_finalize_phase_done = CycleStatusUtils.is_finalize_phase_done(cycle_dir)
            if is_training_phase_done and not is_finalize_phase_done:
                return get_best_ckpt_from_dir(current_dir, cycle_idx), cycle_idx

        # if non_final_phase_cycle is not None:
        #     return get_best_ckpt_from_dir(current_dir, non_final_phase_cycle), non_final_phase_cycle

        return None, None
    else:
        blobs = bucket.list_blobs(prefix=f"{current_dir}/")
        for blob in blobs:
            # blobs contains all of the files in the current_dir, including those in subdirectories
            print(blob.name, file=sys.stderr)
            path_parts = blob.name.split('/')
            for part in path_parts:
                if "cycle_" in part:
                    cycle_idx = part.split("_")[-1]
                    if cycle_idx.isdigit():
                        last_cycle = max(last_cycle, int(cycle_idx))

        # non_final_phase_cycle = None
        # for cycle_idx in range(last_cycle, -1, -1):
        for cycle_idx in range(0, last_cycle + 1):
            cycle_dir = f"{current_dir}/cycle_{cycle_idx}"
            is_training_phase_done = CycleStatusUtils.is_training_phase_done(cycle_dir)
            is_finalize_phase_done = CycleStatusUtils.is_finalize_phase_done(cycle_dir)
            if is_training_phase_done and not is_finalize_phase_done:
                blobs = bucket.list_blobs(prefix=f"{cycle_dir}/")
                for blob in blobs:
                    if "epoch" in blob.name and blob.name.endswith(".ckpt"):
                        # return f"gs://generative_active_learning/{blob.name}", cycle_idx
                        return f"gs://{bucket_name}/{blob.name}", cycle_idx
        
        return None, None





def get_model_checkpoint_path_from_default_root_dir(default_root_dir):
    """
    Get the model checkpoint path from the default root directory.
    
    Args:
        default_root_dir (str): Default root directory containing the model checkpoint.
    
    Returns:
        (str)
    """

    return default_root_dir


def get_model_checkpoint_callback(trainer):
    """
    Get the model checkpoint callback from the trainer's callbacks.
    
    Args:
        trainer (L.Trainer): Lightning Trainer instance.
        
    Returns:
        L.pytorch.callbacks.ModelCheckpoint: Model checkpoint callback if available, otherwise None.
    """
    for callback in trainer.callbacks:
        if isinstance(callback, L.pytorch.callbacks.ModelCheckpoint):
            return callback
    return None

def get_best_metrics_from_trainer(trainer):
    """
    Get the model checkpoint path from the trainer's callbacks.
    
    Args:
        trainer (L.Trainer): Lightning Trainer instance.
        
    Returns:
        str: Path to the model checkpoint if available, otherwise None.
    """
    for callback in trainer.callbacks:
        if isinstance(callback, L.pytorch.callbacks.ModelCheckpoint):
            return {callback.monitor: callback.best_model_score}
    return None

def get_callbacks(cfg, default_root_dir, cycle_idx=0, train_until_final_phase=False, is_done=False):
    callback_cfg = cfg.callbacks if 'callbacks' in cfg else {}
    callbacks = []

    model_checkpoint_cfg = callback_cfg.model_checkpoint
    base_model_path = get_model_checkpoint_path_from_default_root_dir(default_root_dir)
    dirpath = f"{base_model_path}/cycle_{cycle_idx}"
    print("dirpath:", dirpath)
    
    if "early_stopping" in callback_cfg:
        early_stopping_cfg = callback_cfg.early_stopping
        callbacks.append(
            L.pytorch.callbacks.EarlyStopping(
                monitor=early_stopping_cfg.monitor if 'monitor' in early_stopping_cfg else 'val_loss',
                patience=early_stopping_cfg.patience if 'patience' in early_stopping_cfg else 5,
                verbose=early_stopping_cfg.verbose if 'verbose' in early_stopping_cfg else False,
                mode=early_stopping_cfg.mode if 'mode' in early_stopping_cfg else 'min'
            )
        )

        # callbacks.append(ResetEarlyStoppingState(cfg.callbacks.early_stopping.get('patience', None)))
        if is_done:
            callbacks.append(ResetEarlyStoppingState(
                cfg.callbacks.early_stopping.get('patience', None), 
                train_until_final_phase=train_until_final_phase))
        # callbacks.append(EarlyStoppingRenameCallback())

    if "model_checkpoint" in callback_cfg:
        filename_format = '{epoch:02d}-{val_accuracy:.2f}' if not train_until_final_phase else 'final-{epoch:02d}-{val_accuracy:.2f}'
        callbacks.append(
            L.pytorch.callbacks.ModelCheckpoint(
                # dirpath=os.path.join(default_root_dir, f"cycle_{cycle_idx}"),
                dirpath=dirpath,
                verbose=model_checkpoint_cfg.get('verbose', False),
                # filename=model_checkpoint_cfg.get('filename', '{epoch:02d}-{val_loss:.2f}'),
                # monitor=model_checkpoint_cfg.get('monitor', 'val_loss'),
                filename=model_checkpoint_cfg.get('filename', filename_format),
                monitor=model_checkpoint_cfg.get('monitor', 'val_accuracy'),
                save_top_k=model_checkpoint_cfg.get('save_top_k', 1),
                save_last=model_checkpoint_cfg.get('save_last', True),
                mode=model_checkpoint_cfg.get('mode', 'min')
            )
        )

        if train_until_final_phase:
            print("Adding ModifyModelCheckpointFormatState callback for final phase.")
            # callbacks.append(ModifyModelCheckpointFormatState(filename_format))
            callbacks.append(ModifyModelCheckpointFormatState(
                dir_path=dirpath, filename_format=filename_format))
    
    if "learning_rate_monitor" in callback_cfg:
        learning_rate_monitor_cfg = callback_cfg.learning_rate_monitor
        callbacks.append(
            L.pytorch.callbacks.LearningRateMonitor(
                logging_interval=learning_rate_monitor_cfg.get('logging_interval', None)
            )
        )
    
    if os.environ.get("PJRT_DEVICE") == "TPU":
        # gcs_uploader_callback = GCSUploaderCallback(
        #     bucket_name="generative_active_learning",
        #     gcs_dir_path=dirpath
        # )
        gcs_uploader_callback = GCSUploaderCallback(
            bucket_name=bucket_name,
            gcs_dir_path=dirpath
        )
        callbacks.append(gcs_uploader_callback)
    return callbacks


def get_trainer(cfg, cycle=0, logger=None, 
                start_datetime=None, seed=None, default_root_dir=None, num_devices=1, 
                train_until_final_phase=False, num_nodes=1, is_done=True):
    """
    Create a Lightning Trainer based on the provided configuration.
    
    Args:
        cfg (DictConfig): Configuration object containing trainer settings.
        
    Returns:
        L.Trainer: Configured Lightning Trainer instance.
    """

    # accelerator = cfg.device.get('device_type', 'auto')
    if os.environ.get("DISABLE_XLA") != "1" and os.environ.get("PJRT_DEVICE") == "TPU":
        accelerator = 'tpu'
    else:
        # accelerator = cfg.device.get('device_type', 'auto')
        accelerator = 'auto'
    
    seed = seed if seed is not None else cfg.seed if 'seed' in cfg else 42

    # run_id = wandb_logger.experiment.id if resume_wandb_run_id is None else resume_wandb_run_id
    assert start_datetime is not None, "start_datetime must be provided for the workspace name."

    val_check_interval = cfg.train.get('val_check_interval', 1.0)
    if cfg.train.get("increasing_val_freq", False):
        val_freq = cycle + 1
        try:
            val_check_interval = val_check_interval / val_freq
            cfg.callbacks.early_stopping.patience = cfg.callbacks.early_stopping.patience * val_freq
            print(f"Patient callback is modified as {cfg.callbacks.early_stopping.patience}", file=sys.stderr)
        except:
            print("Warning: `increasing_val_freq` is set, the early stopping patient is not set. ", file=sys.stderr)
        finally:
            print(f"Val check interval is set as {val_check_interval:.3f}", file=sys.stderr)
            
    if os.environ.get("PJRT_DEVICE") == "TPU" and isinstance(val_check_interval, float) and val_check_interval != 1.0:
        num_initial_labeled_generated_samples = cfg.num_initial_labeled_generated_samples
        num_generated_sample_per_cycle = cfg.generative_oracle.num_generated_sample_per_cycle
        total_num_generated_samples = num_initial_labeled_generated_samples + num_generated_sample_per_cycle * cycle
        val_check_interval = int(total_num_generated_samples / cfg.train.batch_size / val_freq)
        print(f"TPU: Val check interval is set as {val_check_interval}", file=sys.stderr)


    callbacks = get_callbacks(cfg, default_root_dir=default_root_dir, cycle_idx=cycle, train_until_final_phase=train_until_final_phase, is_done=is_done)
    gradient_accumulation_steps = cfg.train.get('gradient_accumulation_steps', 1)

    if cfg.train.get('mixed_precision', False):
        if os.environ.get("PJRT_DEVICE") == "TPU":
            # precision = 16
            # precision = "bf16-true"
            precision = 32 # the targetmodelframework will be account for mixed precision.
        else:
            precision = 'bf16-mixed'
    else:
        precision = 32
    
    gradient_clip_val = cfg.train.get('gradient_clip_val', None)
    print("gradient_clip_val:", gradient_clip_val)

    max_steps = cfg.train.get("max_steps", -1)
    if max_steps > 0:
        print("max_steps:", cfg.train.get("max_steps", -1))
        callbacks.append(CustomMaxStepsCallback(max_steps))

    strategy = "auto"

    trainer = L.Trainer(
        # max_epochs=cfg.train.get("max_epochs", 1000),
        max_epochs=cfg.train.get("max_epochs", -1),
        # max_steps=cfg.train.get("max_steps", -1),
        accelerator=accelerator,
        devices=num_devices if not _USE_XLA else "auto",
        strategy=strategy,
        num_nodes=num_nodes,
        logger=logger,
        log_every_n_steps=50,
        profiler=cfg.profiler if 'profiler' in cfg else "simple",
        default_root_dir=default_root_dir,
        callbacks=callbacks,
        deterministic=cfg.deterministic if 'deterministic' in cfg else True,
        accumulate_grad_batches=gradient_accumulation_steps,
        val_check_interval=val_check_interval,
        # val_check_interval=100, # TMP
        check_val_every_n_epoch=cfg.train.get('check_val_every_n_epoch', 1),
        precision=precision,
        gradient_clip_val=gradient_clip_val,
    )
    return trainer, default_root_dir, logger
