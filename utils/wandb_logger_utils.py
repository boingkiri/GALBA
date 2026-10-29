import os
import yaml

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.loggers import CSVLogger

try:
    import torch_xla
    from google.cloud import storage
    _USE_XLA = True
    storage_client = storage.Client()
    # bucket = storage_client.bucket("generative_active_learning")
    GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "generative_active_learning")
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
except ImportError:
    _USE_XLA = False


def get_project_name(cfg):
    # return cfg.project_name if 'project_name' in cfg else "default_project"
    return f"{cfg.project_name}_{cfg.dataset.name}" if 'project_name' in cfg else "default_project"

def get_group_name(cfg):
    return cfg.exp_name if 'exp_name' in cfg else "default_experiment"


def create_logger(cfg, seed, default_root_dir, accelerator=None, train_until_final_phase=False):
    # Initialize logger on main process first, and then get the logger for the rest of the process
    if accelerator is not None:
        if accelerator.is_main_process:
            logger = create_logger_per_process(cfg, seed, default_root_dir, train_until_final_phase=train_until_final_phase)
        accelerator.wait_for_everyone()
        if not accelerator.is_main_process:
            logger = create_logger_per_process(cfg, seed, default_root_dir, train_until_final_phase=train_until_final_phase)
    else:
        logger = create_logger_per_process(cfg, seed, default_root_dir, train_until_final_phase=train_until_final_phase)
    return logger

def create_logger_per_process(cfg, seed, default_root_dir, train_until_final_phase=False):
    if cfg.logger_type == "wandb":
        return create_wandb_logger(cfg, seed, default_root_dir, train_until_final_phase=train_until_final_phase)
    elif cfg.logger_type == "csv":
        return create_csv_logger(cfg, seed, default_root_dir, train_until_final_phase=train_until_final_phase)
    else:
        raise ValueError(f"Unknown logger type: {cfg.logger_type}")

def create_wandb_logger(cfg, seed, default_root_dir, train_until_final_phase=False):
    """
    Create a WandbLogger instance for logging.
    
    Args:
        cfg (DictConfig): Configuration dictionary.
        cycle_idx (int): Current cycle index.
        start_datetime (str): Start datetime string.
        seed (int): Random seed for reproducibility.
        
    Returns:
        WandbLogger: Configured WandbLogger instance.
    """

    yaml_file_name = "wandb_config.yaml" if not train_until_final_phase else "wandb_config_final_phase.yaml"
    if not _USE_XLA:
        if yaml_file_name in os.listdir(default_root_dir):
            with open(os.path.join(default_root_dir, yaml_file_name), 'r') as f:
                wandb_config = yaml.load(f, Loader=yaml.FullLoader)
                wandb_logger = WandbLogger(**wandb_config)

        else:
            exp_name = f"{get_group_name(cfg)}_seed_{seed}" if not train_until_final_phase else f"{get_group_name(cfg)}_final_phase_seed_{seed}"
            wandb_logger_args = {
                "project": get_project_name(cfg),
                "group": get_group_name(cfg),
                "name": exp_name,
                "config": {"seed": seed},
            }
            wandb_logger = WandbLogger(**wandb_logger_args)
            wandb_logger_run_id = wandb_logger.experiment.id
            wandb_logger_args["id"] = wandb_logger_run_id

            # with open(os.path.join(default_root_dir, "wandb_config.yaml"), 'w') as f:
            #     yaml.dump(wandb_logger_args, f)
            with open(os.path.join(default_root_dir, yaml_file_name), 'w') as f:
                yaml.dump(wandb_logger_args, f)
    else:
        if yaml_file_name in [blob.name.split("/")[-1] for blob in bucket.list_blobs(prefix=default_root_dir)]:
            blob = bucket.blob(os.path.join(default_root_dir, yaml_file_name))
            yaml_content = blob.download_as_string()
            wandb_config = yaml.load(yaml_content, Loader=yaml.FullLoader)
            wandb_logger = WandbLogger(**wandb_config)
        else:
            exp_name = f"{get_group_name(cfg)}_seed_{seed}" if not train_until_final_phase else f"{get_group_name(cfg)}_final_phase_seed_{seed}"
            wandb_logger_args = {
                "project": get_project_name(cfg),
                "group": get_group_name(cfg),
                "name": exp_name,
                "config": {"seed": seed},
            }
            wandb_logger = WandbLogger(**wandb_logger_args)
            wandb_logger_run_id = wandb_logger.experiment.id
            wandb_logger_args["id"] = wandb_logger_run_id
            
            blob = bucket.blob(os.path.join(default_root_dir, yaml_file_name))
            blob.upload_from_string(yaml.dump(wandb_logger_args))

    return wandb_logger


def create_csv_logger(cfg, seed, default_root_dir):
    # csv_logger = CSVLogger(
    #     save_dir=default_root_dir,
    #     name=f"seed_{seed}",
    #     version=cfg.get("version", 1),
    # )
    csv_logger = CSVLogger(
        save_dir=default_root_dir,
        # name=f"seed_{seed}",
        name=f"logging",
        version=cfg.get("version", 1),
    )
    return csv_logger