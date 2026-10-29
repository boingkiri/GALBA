import sys
import torch
import torch.nn as nn
import torchmetrics
import torchvision

import numpy as np
import lightning as L

from .target_model_container import TargetModelContainer

from copy import deepcopy

# import Omegaconf
# from Omegaconf import OmegaConf
from omegaconf import OmegaConf

from functools import partial
from contextlib import nullcontext

try:
    import torch_xla
    from torch_xla.amp import syncfree
    _HAS_XLA = True
except Exception:
    _HAS_XLA = False


class TargetModelFramework(L.LightningModule):
# class TargetModelFramework(BNCompatLightningMixin, L.LightningModule):
    bn_target_on_save = "batch"        # export for GPU as SyncBN
    bn_target_on_load_gpu = "sync"    # on GPU, keep SyncBN
    def __init__(self, cfg, cycle=0):
        super().__init__()

        self.save_hyperparameters()

        if isinstance(cfg, dict):
            # self.cfg = L.core.LightningConfig(cfg)
            cfg = OmegaConf.create(cfg)


        self.cfg = cfg
        self.cycle = cycle
        self.task = cfg.target_model.get("task", "classification")
        self.target_model = TargetModelContainer(cfg)

        # Add manual xlasyncbatch when using TPU
        # if _HAS_XLA:
        #     convert_bn_to_xla_dimaware(self.target_model)

        # self.post_transform = self.inflate_post_transform()

        # self.use_learning_loss_during_training  = cfg.target_model.loss_module.get("use_learning_loss_during_training", True)
        # self.automatic_optimization = False

        if cfg.train.get("ema", {}).get("enabled", False):
            self.ema_model = deepcopy(self.target_model).eval()
            self.ema_decay = self.cfg.train.ema.decay
            self.ema_update_after_step = cfg.train.ema.update_after_step

            for p in self.ema_model.parameters():
                p.requires_grad = False
                # p.requires_grad_(False)
            self._last_ema_global_step = -1
        
        self.monitor_cfg = self.cfg.callbacks.model_checkpoint # or self.cfg.callbacks.early_stopping, whatever
        self.monitor_key = self.monitor_cfg.monitor # or self.cfg.callbacks.early_stopping.monitor, whatever
        self.monitor_mode = self.monitor_cfg.mode
        self.best_metric = float("-inf") if self.monitor_mode == "max" else float("inf")
        self.xla_mixed_precision_context = partial(torch.autocast, "xla") if _HAS_XLA and cfg.train.get('mixed_precision', False) else nullcontext

    def on_fit_start(self):
        if hasattr(self, "ema_model"):
            dev = next(self.target_model.parameters()).device
            dtype = next(self.target_model.parameters()).dtype
            self.ema_model.to(device=dev, dtype=dtype)

            print("EMA model dtype:", next(self.ema_model.parameters()).dtype)


    def forward(self, x):
        # Use the ema model if it exists, otherwise use the target model
        # forward_model = self.ema_model if hasattr(self, 'ema_model') else self.target_model
        # return forward_model(x)
        return self.target_model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        # images, labels = self.post_transform(images, labels)
        # out, features = self.target_model.forward_with_features(images)
        with self.xla_mixed_precision_context():
            out = self.target_model.forward(images)
            # Compute the loss for the target model
            loss_structure = self.target_model.loss_function(out.float(), labels)


        loss = loss_structure["loss"]
        loggings = loss_structure["logging"]

        if torch.isnan(loss):
            self.trainer.save_checkpoint(f"nan_error_checkpoint_epoch_{self.trainer.current_epoch}.ckpt")
            torch.save(images.cpu(), f"nan_error_images_epoch_{self.trainer.current_epoch}_{self.global_rank}.pt")
            torch.save(labels.cpu(), f"nan_error_labels_epoch_{self.trainer.current_epoch}_{self.global_rank}.pt")
            raise ValueError("NaN loss encountered during training.")
        # aux = loss_structure.get("aux", {})

        for logging in loggings:
            # self.log(f'Cycle_{self.cycle}/train_{logging}', loggings[logging], on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log(f'Cycle_{self.cycle}/train_{logging}', loggings[logging], on_step=True, on_epoch=True, logger=True)
        self.log(f'Cycle_{self.cycle}/train_total_loss', loss, prog_bar=True)

        return loss

    # @torch.no_grad()
    # def _ema_update(self, model, ema_model, decay: float):
    #     for (name, p), (_, ema_p) in zip(model.named_parameters(), ema_model.named_parameters()):
    #         ema_p.mul_(decay).add_(p, alpha=1 - decay)
    #     # BN 등 버퍼 동기화(매 스텝 또는 주기적으로)
    #     for b, ema_b in zip(model.buffers(), ema_model.buffers()):
    #         ema_b.copy_(b)

    @torch.no_grad()
    def _ema_update(self, decay=None, bn_buffers_policy="copy"):
        """
        bn_buffers_policy: "copy" | "ema" | "ignore"
            - "copy": copy running_mean/var from live model (recommended default)
            - "ema" : EMA them too (only for very small batch per device)
            - "ignore": leave BN running stats unchanged (then do recalibration later)
        """
        if decay is None:
            decay = self.ema_decay

        msd = self.target_model.state_dict()
        esd = self.ema_model.state_dict()

        for k, ema_v in esd.items():
            v = msd[k]
            is_bn_buf = ("running_mean" in k) or ("running_var" in k)
            if not torch.is_floating_point(v):
                # e.g., num_batches_tracked
                ema_v.copy_(v)
                continue

            if is_bn_buf:
                if bn_buffers_policy == "copy":
                    ema_v.copy_(v)
                elif bn_buffers_policy == "ema":
                    ema_v.mul_(decay).add_(v, alpha=1.0 - decay)
                # "ignore" -> do nothing; plan to recalibrate later
            else:
                # trainable params, incl. BN affine weight/bias
                ema_v.mul_(decay).add_(v, alpha=1.0 - decay)


    # --- gradient-accumulation aware EMA: only when an optimizer step occurred
    def on_train_batch_end(self, outputs, batch, batch_idx):
        if not hasattr(self, "ema_model"):
            return

        # when grad accumulation is used, global_step only increments on optimizer step
        gs = int(self.global_step)  # Lightning increments after the step
        if gs == self._last_ema_global_step:
            # no optimizer step this batch -> skip EMA update
            return

        # optimizer step happened -> maybe update EMA (respect warmup)
        if gs >= getattr(self, "ema_update_after_step", 0):
            self._ema_update()
            self._last_ema_global_step = gs

            if _HAS_XLA:
                torch_xla.sync()

    def on_after_backward(self):
        # Log of the norm of the all of parameters 
        if self.trainer.global_step % 5 == 0:  # Log every 5 steps
            grad_norm_val = 0
            for name, param in self.named_parameters():
                if param.grad is not None:
                    grad_norm_val += param.grad.norm()
            grad_norm_val = grad_norm_val.detach()
            self.log(f"gradients/norm", grad_norm_val, prog_bar=False, logger=True)

    def predict(self, batch):
        images, _ = batch
        # out, _ = self.target_model.forward_with_features(images)
        out = self.target_model.forward(images)
        return out

    def loss_function(self):
        return self.target_model.loss_function
    
    def loss_function(self, preds, targets):
        return self.target_model.loss_function(preds, targets)

    def validation_step(self, batch, batch_idx):
        images, labels = batch

        validation_model = self.ema_model if hasattr(self, 'ema_model') else self.target_model
        # validation_model = self.target_model # TMP
        
        # out, features = self.target_model.forward_with_features(images)
        validation_model.eval()
        out = validation_model.forward(images)

        if self.task == "classification":
            target_model_loss = nn.CrossEntropyLoss()(out, labels)

            validation_metrics = self.target_model.measure_metrics(out, labels)
            for metric, value in validation_metrics.items():
                self.log(f'Cycle_{self.cycle}/val_{metric}', value, on_epoch=True, sync_dist=True)
                self.log(f"val_{metric}", value, logger=False, on_epoch=True, sync_dist=True, prog_bar=True)
        else:
            raise ValueError(f"Task {self.task} is not supported for validation metrics.")

        # loss = target_model_loss + learning_loss_loss
        loss = target_model_loss
        self.log(f'Cycle_{self.cycle}/val_target_model_loss',
                 target_model_loss.to(self.device) if isinstance(target_model_loss, torch.Tensor) else target_model_loss,
                 on_epoch=True, sync_dist=True)

        validation_model.train()
        # metric 
        return loss
    
    
    def on_validation_epoch_end(self):
        # 이번 에폭의 validation 집계 결과에서 현재 값 읽기
        if self.monitor_key in self.trainer.callback_metrics:
            cur = self.trainer.callback_metrics[self.monitor_key]
            try:
                cur = float(getattr(cur, "detach", lambda: cur)().cpu().item())
            except Exception:
                cur = float(cur)

            improved = (cur > self.best_metric) if self.monitor_mode == "max" else (cur < self.best_metric)
            if improved:
                self.best_metric = cur
                self.log_dict(
                    {f"best_{self.monitor_key}": self.best_metric, "cycle": self.cycle},
                    on_epoch=True, logger=False, prog_bar=False, sync_dist=True
                )
            
    def get_optimizer(self, parameters):
        optimizer_cfg = self.cfg.train.optimizer
        # On TPU, use the sync-free optimizers from torch_xla.
        optim_module = syncfree if _HAS_XLA else torch.optim
        if optimizer_cfg.name == 'adam':
            optimizer = optim_module.Adam(
                parameters,
                lr=optimizer_cfg.lr,
                weight_decay=optimizer_cfg.weight_decay,
                betas=optimizer_cfg.get("betas", (0.9, 0.999)),
                eps=optimizer_cfg.get("eps", 1e-8))
        elif optimizer_cfg.name == 'adamw':
            optimizer = optim_module.AdamW(
                parameters,
                lr=optimizer_cfg.lr,
                weight_decay=optimizer_cfg.get("weight_decay", 0.01),
                betas=optimizer_cfg.get('betas', (0.9, 0.999)),
                eps=optimizer_cfg.get("eps", 1e-8))
        elif optimizer_cfg.name == 'sgd':
            optimizer = torch.optim.SGD(
                parameters,
                lr=optimizer_cfg.lr,
                momentum=optimizer_cfg.get('momentum', 0.9),
                weight_decay=optimizer_cfg.get("weight_decay", 0),
                nesterov=optimizer_cfg.get("nesterov", False))
        else:
            raise NotImplementedError(f"Optimizer name {optimizer_cfg.name} is not implemented yet.")
        return optimizer

    def get_lr_scheduler(self, optimizer, scheduler_cfg=None):
        lr_scheduler_cfg = self.cfg.train.lr_scheduler if scheduler_cfg is None else scheduler_cfg
        if lr_scheduler_cfg.name in ['linear', 'LinearLR']:
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, 
                start_factor=lr_scheduler_cfg.get("start_factor", 1.0),
                end_factor=lr_scheduler_cfg.get("end_factor", 0.0),
                total_iters=lr_scheduler_cfg.get("total_iters", 100))
        elif lr_scheduler_cfg.name in ['step', 'StepLR']:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=lr_scheduler_cfg.get("step_size", 100), 
                gamma=lr_scheduler_cfg.get("gamma", 0.1))
        elif lr_scheduler_cfg.name in ['constant', 'ConstantLR']:
            scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, 
                factor=lr_scheduler_cfg.get("factor", 1.0),
                total_iters=lr_scheduler_cfg.get("total_iters", 100))
        elif lr_scheduler_cfg.name in ['exponential', 'ExponentialLR']:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, 
                gamma=lr_scheduler_cfg.get("gamma", 0.99))
        elif lr_scheduler_cfg.name in ['multi_step', 'MultiStepLR', 'multistep_lr']:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, 
                milestones=lr_scheduler_cfg.get("milestones", []), 
                gamma=lr_scheduler_cfg.get("gamma", 0.1))
        elif lr_scheduler_cfg.name in ['cosine', 'CosineAnnealingLR']:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=lr_scheduler_cfg.get("T_max", 100),
                eta_min=lr_scheduler_cfg.get("eta_min", 0.0))
        elif lr_scheduler_cfg.name in ["sequential", "SequentialLR"]:
            schedulers = []
            for sub_scheduler_cfg in lr_scheduler_cfg.schedulers:
                sub_scheduler = self.get_lr_scheduler(optimizer, sub_scheduler_cfg)
                schedulers.append(sub_scheduler)
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, 
                schedulers=schedulers, 
                milestones=lr_scheduler_cfg.milestones if 'milestones' in lr_scheduler_cfg else [])
        else:
            raise NotImplementedError(f"Learning rate name {lr_scheduler_cfg.name} is not implemented yet.")
        return scheduler

    def configure_optimizers(self):
        target_model_parameters = self.target_model.parameters()
        target_model_optimizer = self.get_optimizer(target_model_parameters)
        target_model_scheduler = self.get_lr_scheduler(target_model_optimizer)
        return {
            "optimizer": target_model_optimizer, 
            "lr_scheduler": {
                "scheduler": target_model_scheduler,
                "interval": self.cfg.train.lr_scheduler.get("interval", "epoch"),
                "frequency": 1
            }
        }


    def set_cycle(self, cycle=None):
        assert cycle is not None
        self.cycle = cycle