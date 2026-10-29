import torch
# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import EarlyStopping
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint

from typing_extensions import override

import os
import shutil

try:
    import torch_xla
    _USE_XLA=True
    from google.cloud import storage
    # bucket = storage.Client().bucket("generative_active_learning")
    GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "generative_active_learning")
    bucket = storage.Client().bucket(GCS_BUCKET_NAME)
except ImportError:
    _USE_XLA=False

class ResetEarlyStoppingState(Callback):
    """
    A custom callback that resets the state of the EarlyStopping callback
    when resuming from a checkpoint.
    """
    def __init__(self, patience: int = None, train_until_final_phase: bool = False):
        super().__init__()
        self.patience = patience
        self.train_until_final_phase = train_until_final_phase

    # @override
    # def on_load_checkpoint(self, trainer, pl_module, checkpoint):
    #     """
    #     Called when a checkpoint is loaded.
    #     """
    #     print("\n--- Custom Callback: on_load_checkpoint called. ---")
    #     print('on-load-checkpoint callback; current-process-idx:', trainer.global_rank)

    @override
    def on_fit_start(self, trainer, pl_module):
        """
        Called when the fit starts.
        """
        early_stopping_callback = None
        for callback in trainer.callbacks:
            if isinstance(callback, EarlyStopping):
                early_stopping_callback = callback
                break
        
        if not early_stopping_callback:
            print("Warning: Could not find an EarlyStopping callback to reset.")
            return

        # Reset the state attributes
        early_stopping_callback.wait_count = 0
        early_stopping_callback.stopped_epoch = 0
        
        if early_stopping_callback.mode == "min":
            # early_stopping_callback.best_score = torch.tensor(torch.finfo(torch.float).max)
            early_stopping_callback.best_score = torch.tensor(float('inf'))
        else:
            # early_stopping_callback.best_score = torch.tensor(torch.finfo(torch.float).min)
            early_stopping_callback.best_score = torch.tensor(float('-inf'))
        print(f"early_stopping_callback.patience: {early_stopping_callback.patience}")

        if self.patience is not None:
            early_stopping_callback.patience = self.patience
            print(f"EarlyStopping patience has been set to {early_stopping_callback.patience}.")
        
        if self.train_until_final_phase:
            early_stopping_callback.patience = float('inf')
            print("EarlyStopping patience has been set to infinity for training until final phase.")
    

        print("\n--- Custom Callback: EarlyStopping state has been reset. ---")
        print('current-process-idx:', trainer.global_rank)

    # --- ADD THESE TWO METHODS TO FIX THE ERROR ---
    @override
    def state_dict(self):
        """
        An empty state_dict to make the callback stateful and pass validation.
        """
        return {}

    @override
    def load_state_dict(self, state_dict):
        """
        An empty load_state_dict to make the callback stateful.
        """
        pass

class ModifyModelCheckpointFormatState(Callback):
    """
    A custom callback that modifies the ModelCheckpoint filename format
    when resuming from a checkpoint.
    """

    def __init__(self, dir_path, filename_format=None):
        super().__init__()
        self.dir_path = dir_path
        self.filename_format = filename_format

    @override
    def on_fit_start(self, trainer, pl_module):
        """
        Called when the fit starts.
        """
        model_checkpoint_callback = None
        for callback in trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):  # Replace with actual ModelCheckpoint class if needed
                model_checkpoint_callback = callback
                break
        
        if not model_checkpoint_callback:
            print("Warning: Could not find a ModelCheckpoint callback to modify.")
            return

        # Modify the filename format
        old_filename = model_checkpoint_callback.filename
        model_checkpoint_callback.filename = f"final-{old_filename}" if self.filename_format is None else self.filename_format

        # Reset the best score
        # model_checkpoint_callback.best_model_score = 

        if trainer.is_global_zero and not _USE_XLA:
            # shutil.copy()
            best_checkpoint_path = model_checkpoint_callback.best_model_path
            duplicated_best_checkpoint_path = "final-" + os.path.basename(best_checkpoint_path)
            parent_dir = self.dir_path
            duplicated_best_checkpoint_full_path = os.path.join(parent_dir, duplicated_best_checkpoint_path)
            best_checkpoint_path = os.path.join(self.dir_path, os.path.basename(best_checkpoint_path))
            shutil.copy(best_checkpoint_path, duplicated_best_checkpoint_full_path)


        print(f"ModelCheckpoint filename has been modified to: {model_checkpoint_callback.filename}")
        print("\n--- Custom Callback: ModelCheckpoint filename format has been finalized. ---")

    @override
    def on_validation_end(self, trainer, pl_module):
        pass

class CustomMaxStepsCallback(Callback):
    def __init__(self, max_steps: int):
        super().__init__()
        self.max_steps = max_steps

    @override
    def on_train_batch_end(self, trainer, *args, **kwargs):
        if trainer.global_step >= self.max_steps:
            trainer.should_stop = True
            print(f"\n--- Custom Callback: Stopping training at max_steps={self.max_steps}. ---")
        

class GCSUploaderCallback(Callback):
    def __init__(self, bucket_name, gcs_dir_path):
        self.bucket_name = bucket_name
        self.gcs_dir_path = gcs_dir_path
        self.client = None
        
        # [NEW] 이전에 업로드한 Best Model의 GCS 경로를 기억하는 변수
        self.prev_best_blob_path = None 

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.global_rank == 0:
            ckpt_callback = trainer.checkpoint_callback
            
            if not ckpt_callback:
                return

            # -------------------------------------------------------
            # 1. Best Model 관리 (새로운게 나오면 이전거 삭제)
            # -------------------------------------------------------
            if ckpt_callback.best_model_path and os.path.exists(ckpt_callback.best_model_path):
                current_local_best = ckpt_callback.best_model_path
                filename = os.path.basename(current_local_best)
                current_blob_path = f"{self.gcs_dir_path}/{filename}"

                # "지금 저장하려는 파일"이 "이전에 저장한 파일"과 이름이 다르면? -> 새로운 Best 갱신됨
                if current_blob_path != self.prev_best_blob_path:
                    
                    # (1) 새로운 Best 업로드
                    self._upload(current_local_best, current_blob_path)
                    
                    # (2) 이전 Best 삭제 (기록된 경로가 있다면)
                    if self.prev_best_blob_path:
                        print(f"[GCS Callback] Cleaning up old best: {self.prev_best_blob_path}")
                        self._delete_blob(self.prev_best_blob_path)
                    
                    # (3) 현재 경로를 '이전 경로'로 업데이트
                    self.prev_best_blob_path = current_blob_path

            # -------------------------------------------------------
            # 2. Last Model 관리 (Resume용, 보통 덮어쓰기 되므로 삭제 불필요)
            # -------------------------------------------------------
            if ckpt_callback.last_model_path and os.path.exists(ckpt_callback.last_model_path):
                # last.ckpt는 이름이 고정이라 GCS에서 자동으로 덮어씌워짐 (삭제 로직 불필요)
                last_filename = os.path.basename(ckpt_callback.last_model_path)
                last_blob_path = f"{self.gcs_dir_path}/{last_filename}"
                self._upload(ckpt_callback.last_model_path, last_blob_path)

    def _upload(self, local_path, blob_path):
        """파일 업로드 헬퍼"""
        if self.client is None: self.client = storage.Client()
        bucket = self.client.bucket(self.bucket_name)
        blob = bucket.blob(blob_path)
        
        # print(f"Uploading to {blob_path}...") # 로그가 너무 많으면 주석 처리
        blob.upload_from_filename(local_path)

    def _delete_blob(self, blob_path):
        """[NEW] GCS 파일 삭제 헬퍼"""
        try:
            if self.client is None: self.client = storage.Client()
            bucket = self.client.bucket(self.bucket_name)
            blob = bucket.blob(blob_path)
            
            if blob.exists():
                blob.delete()
                print(f"[GCS Callback] Deleted old checkpoint: {blob_path}")
        except Exception as e:
            print(f"[GCS Callback] Warning: Failed to delete {blob_path}. Reason: {e}")
