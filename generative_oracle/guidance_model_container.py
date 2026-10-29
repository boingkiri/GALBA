import os
from abc import ABC, abstractmethod 

import torch
from torch import flatten
import torchvision

import sys

import torch
import torchvision
import torch.distributed as dist
from torch.func import functional_call
from torch.autograd import grad as autograd_grad
from torch.autograd.functional import hessian
import einops

from tqdm import tqdm
import contextlib

from transformers import CLIPProcessor, CLIPModel
from functools import partial
import typing

import gc

from utils.data_utils import get_class_idx_to_class_name_mapping

from PIL import Image

from torch.utils.checkpoint import checkpoint
from contextlib import nullcontext

try:
    import torch_xla
    _USE_XLA = True
except ImportError:
    _USE_XLA = False

def process_image(image):
    image = (image / 2 + 0.5)
    image = image - image.min().detach()
    image = image / image.max().detach()
    return image

def imagenet_normalization(images):
    """
    Normalize images using ImageNet statistics.

    Args:
        images: A batch of images as a tensor of shape (B, C, H, W) with pixel values in [0, 1].

    Returns:
        Normalized images.
    """
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
    normalized_images = (images - imagenet_mean) / imagenet_std
    return normalized_images


class GuidanceModelContainer:
    """
    A container for the guidance model, which is used to generate guidance for the generative oracle.
    """

    def __init__(self, 
                 cfg, accelerator, diffusers_pipe=None, 
                 *args, **kwargs):
        """
        Initializes the GuidanceModelContainer with a given guidance model.

        Args:
            guidance_model: The guidance model to be used for generating guidance.
        """
        self.cfg = cfg
        self.args = args
        self.kwargs = kwargs
        self.accelerator = accelerator
        self.diffusers_pipe = diffusers_pipe
        self.guidance_model = self.create_guidance_model()

        self.use_gradient_checkpointing = self.cfg.guidance_model.get("use_gradient_checkpointing", False)
        if self.use_gradient_checkpointing:
            self.get_output_for_grad = partial(
                checkpoint, self.guidance_model.get_output_for_grad, 
                use_reentrant=False,)
        else:
            self.get_output_for_grad = self.guidance_model.get_output_for_grad


    def create_guidance_model(self):
        """
        Creates a guidance model based on the configuration.

        Args:
            cfg: The configuration object containing parameters for the guidance model.

        Returns:
            An instance of the guidance model.
        """
        # This method should be implemented to create and return the actual guidance model
        # raise NotImplementedError("This method should be implemented in subclasses.")

        print("guidance model type: ", self.cfg.guidance_model.guidance_model_type)
        if self.cfg.guidance_model.guidance_model_type == "no_guidance":
            return NoGuidanceModel()
        elif self.cfg.guidance_model.guidance_model_type == "entropy":
            return EntropyGuidanceModel(self.cfg, accelerator=self.accelerator, diffusers_pipe=self.diffusers_pipe)
        elif self.cfg.guidance_model.guidance_model_type == "predictive_entropy":
            return PredictiveEntropyGuidanceModel(
                cfg=self.cfg, accelerator=self.accelerator, 
                diffusers_pipe=self.diffusers_pipe, 
                *self.args, **self.kwargs)
        elif self.cfg.guidance_model.guidance_model_type == "bald":
            return BALDGuidanceModel(
                cfg=self.cfg, accelerator=self.accelerator, 
                diffusers_pipe=self.diffusers_pipe, 
                *self.args, **self.kwargs)
        else:
            raise ValueError(f"Unsupported guidance model type: {self.cfg.guidance_model.guidance_model_type}")
    
    # def load_guidance_model(self, current_cycle_dir):
    def load_guidance_model(self, checkpoint_path):
        self.guidance_model.load_model(checkpoint_path)

    def value_and_grad_of_model(self, *args, **kwargs):
        """
        Generates guidance based on the input data using the guidance model.

        Args:
            diffusion_model: The diffusion model used to generate the output.
            diffusion_model_input: The input data for the diffusion model.
            diffusion_model_type: The type of the diffusion model (e.g., denoiser,
                epsilon).
            diffusion_model_noisy_input: The noisy input data for the diffusion model.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The generated guidance.
        """

        diffusion_model = kwargs.get("diffusion_model", None)
        diffusion_model_input = kwargs.get("diffusion_model_input", None)
        diffusion_model_type = kwargs.get("diffusion_model_type", None) # denoiser, epsilon

        if self.kwargs.get("dummy_flag", False):
            diffusion_output = diffusion_model(**diffusion_model_input)
            return diffusion_output, torch.zeros_like(diffusion_output), torch.zeros_like(diffusion_output)
        
        with torch.enable_grad():
            kwargs["diffusion_model_noisy_input"].requires_grad_(True) # Ensure the noisy input requires gradients
            diffusion_output = diffusion_model(**diffusion_model_input)

            new_kwargs = kwargs.copy()
            new_kwargs["diffusion_model_output"] = diffusion_output
            new_kwargs["diffusion_model_type"] = diffusion_model_type
            new_kwargs["diffusion_model_class_labels"] = diffusion_model_input["class_labels"]
            # new_kwargs["diffusion_model_noisy_input"] = diffusion_model_input["x"] # TMP
            # grad_of_guidance_model = self.get_grad_of_guidance_model(*args, **new_kwargs)
            grad, aux_output = self.get_grad_of_guidance_model(*args, **new_kwargs)
        
        return diffusion_output, grad, aux_output

    def get_grad_of_guidance_model(self, *args, **kwargs):
        """
        Generates guidance based on the input data using the guidance model.

        Args:
            input_data: The input data for which guidance is to be generated.

        Returns:
            The generated guidance.
        """
        # def get_gpu_memory_usage():
        #     # 현재 할당된 GPU 메모리 (Byte 단위)
        #     return torch.cuda.memory_allocated()
        # mem_before = get_gpu_memory_usage()
        with torch.enable_grad():
            diffusion_model_noisy_input = kwargs.get("diffusion_model_noisy_input", None)
            if diffusion_model_noisy_input is None:
                raise ValueError("diffusion_model_noisy_input is required to compute gradients.")
            diffusion_model_noisy_input.requires_grad_(True)  # Ensure the noisy input requires gradients
            model_output_for_grad, aux_output = self.guidance_model.get_output_for_grad(*args, **kwargs)
            # mem_after = get_gpu_memory_usage()
            # print(f"Guidance model gradient computation GPU memory usage: {(mem_after - mem_before) / (1024 * 1024):.2f} MB", file=sys.stderr)
            if model_output_for_grad is not None:
                grad = torch.autograd.grad(
                outputs=model_output_for_grad,
                inputs=diffusion_model_noisy_input,)[0]
            if model_output_for_grad is None:
                # raise ValueError("model_output_for_grad is None. Check the guidance model implementation.")
                grad = torch.zeros_like(diffusion_model_noisy_input)
        if self.cfg.generative_oracle.get("gradient_rescale_to_denoised_norm", True):
            denom = torch.norm(grad, 2, dim=tuple(range(1, len(grad.shape))), keepdim=True).detach() + 1e-8
            scale = diffusion_model_noisy_input.norm(2, dim=tuple(range(1, len(diffusion_model_noisy_input.shape))), keepdim=True).detach()
            grad = grad / denom * scale
            # print("Rescaled guidance model gradient to match denoised input norm.", file=sys.stderr)
        # else:
        #     print("No rescaling of guidance model gradient.", file=sys.stderr)

        # mem_after = get_gpu_memory_usage()
        # print(f"Guidance model gradient computation GPU memory usage: {(mem_after - mem_before) / (1024 * 1024):.2f} MB", file=sys.stderr)

        if self.cfg.guidance_model.get("plus_direction", False):
            grad = -grad # For gradient ascent
        
        return grad, aux_output

    def get_output_of_guidance_model(self, *args, **kwargs):
        """
        Generates guidance based on the input data using the guidance model.

        Args:
            input_data: The input data for which guidance is to be generated.

        Returns:
            The generated guidance.
        """
        model_output, aux_output = self.guidance_model.get_output_for_grad(*args, **kwargs) # model output for maximize
        return model_output, aux_output

    def get_output(self, eval_inputs, eval_labels=None):
        return self.guidance_model.get_output(eval_inputs, eval_labels)

class GuidanceModel(ABC):
    def __init__(self):
        pass

    # @abstractmethod
    # def get_scaled_input(self, noisy_input, noise_level, noise_pred, timestep):
    #     pass

    @abstractmethod
    def get_output_for_grad(self, *args, **kwargs):
        """
        Generate guidance based on the input data.

        Args:
            input_data: The input data for which guidance is to be generated.
        
        Returns:
            The generated guidance.
        """
        pass

    @abstractmethod
    def get_output(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def load_model(self, checkpoint_path):
        """
        Load the guidance model from a checkpoint or other source.
        
        Args:
            checkpoint_path: The path to the checkpoint file.
        """
        pass

    
    def inflate_pre_transform(self, transform_elem_cfg):
        if "random_horizontal_flip" in transform_elem_cfg:
            return torchvision.transforms.RandomHorizontalFlip(
                p=transform_elem_cfg.random_horizontal_flip.p if 'p' in transform_elem_cfg.random_horizontal_flip else 0.5
            )
        elif "random_crop" in transform_elem_cfg:
            random_crop_cfg = transform_elem_cfg.random_crop
            return torchvision.transforms.RandomCrop(
                random_crop_cfg.size, 
                padding=random_crop_cfg.padding if 'padding' in random_crop_cfg else 0)
        
        elif "resize" in transform_elem_cfg:
            resize_cfg = transform_elem_cfg.resize

            interpolation_mode = torchvision.transforms.InterpolationMode.BILINEAR
            if 'interpolation' in resize_cfg:
                if resize_cfg.interpolation == "bilinear":
                    interpolation_mode = torchvision.transforms.InterpolationMode.BILINEAR
                else:
                    raise ValueError(f"Interpolation mode {resize_cfg.interpolation} is not supported.")
            return torchvision.transforms.Resize(
                size=resize_cfg.size if 'size' in resize_cfg else (32, 32),
                interpolation=interpolation_mode
            )
        elif "center_crop" in transform_elem_cfg:
            center_crop_cfg = transform_elem_cfg.center_crop
            return torchvision.transforms.CenterCrop(
                size=center_crop_cfg.size if 'size' in center_crop_cfg else (32, 32)
            )
        elif "random_resized_crop" in transform_elem_cfg:
            random_resized_crop_cfg = transform_elem_cfg.random_resized_crop
            return torchvision.transforms.RandomResizedCrop(
                size=random_resized_crop_cfg.size if 'size' in random_resized_crop_cfg else (32, 32),
                scale=random_resized_crop_cfg.scale if 'scale' in random_resized_crop_cfg else (0.08, 1.0),
                ratio=random_resized_crop_cfg.ratio if 'ratio' in random_resized_crop_cfg else (3. / 4., 4. / 3.)
            )
        elif "normalize" in transform_elem_cfg:
            normalize_cfg = transform_elem_cfg.normalize
            return torchvision.transforms.Normalize(
                mean=normalize_cfg.mean if 'mean' in normalize_cfg else (0.5, 0.5, 0.5),
                std=normalize_cfg.std if 'std' in normalize_cfg else (0.5, 0.5, 0.5)
            )
        else:
            print(f"Transform {transform_elem_cfg} is not supported.")
            return None

######
######

class ModelInputScaler():
    def __init__(self, time_dependent, noise_scheduler_type, diffusion_model_type):
        self.time_dependent = time_dependent
        self.noise_scheduler_type = noise_scheduler_type
        self.diffusion_model_type = diffusion_model_type
    
    def get_scaled_input(self, noisy_input, diffusion_output, noise_level=None, alphas_cumprod=None):
        """
        Get the scaled input for the model based on the noisy input, diffusion output, and other parameters.
        
        Args:
            noisy_input: The noisy input data.
            diffusion_output: The output from the diffusion model.
            noise_level: The level of noise in the input data (optional).
            noise_pred: The predicted noise (optional).
            timestep: The current timestep in the diffusion process (optional).
        
        Returns:
            The scaled input for the guidance model.
        """
        if self.time_dependent:
            return noisy_input
        
        # If the model is time-independent, we assume the output is the predicted x0 or noise
        if self.diffusion_model_type == "denoiser":
            # For denoiser, we assume the output is the predicted x0
            return diffusion_output

        # If the model is time-independent and the output is epsilon (predicted noise)        
        if self.noise_scheduler_type == "ddpm":
            # For VPSDE, we assume the output is the predicted noise
            if alphas_cumprod is None:
                raise ValueError("alpha_accumulate is required for VPSDE.")
            # return noisy_input - diffusion_output * noise_level / alpha_accumulate
            pred_x0 = (noisy_input - diffusion_output * torch.sqrt(1 - alphas_cumprod)) / torch.sqrt(alphas_cumprod)
            return pred_x0
            # pred_x0 = (noisy_input - diffusion_output * torch.sqrt(1 - alphas_cumprod)) / torch.sqrt(alphas_cumprod)
            # fac = torch.sqrt(1 - alphas_cumprod)
            # sample = pred_x0 * fac + noisy_input * (1 - fac)
            # return sample
        elif self.noise_scheduler_type == "edm":
            # For EDM, we assume the output is the predicted noise
            if noise_level is None:
                raise ValueError("noise_level is required for EDM.")
            return noisy_input - diffusion_output * noise_level
        elif self.noise_scheduler_type == "epsilon":
            pass


######
######
######


class EntropyGuidanceModel(GuidanceModel):
    def __init__(self, cfg, accelerator, diffusers_pipe=None, *args, **kwargs):
        self.cfg = cfg
        self.accelerator = accelerator
        self.diffusers_pipe = diffusers_pipe
        from models.target_model_framework import TargetModelFramework
        self.model_class = TargetModelFramework
        self.scaler_input = ModelInputScaler(
            time_dependent=False,
            noise_scheduler_type=cfg.generative_oracle.noise_scheduler_type,
            diffusion_model_type=cfg.generative_oracle.diffusion_model_type
        )
        self.count = 0

        if self.cfg.generative_oracle.dataset_name != "cifar10":
            def post_process_fn(image):
                image = process_image(image)
                image = imagenet_normalization(image)
                return image
            self.post_process = post_process_fn


    def load_model(self, ckpt_path):
        if ckpt_path is None:
            print("No checkpoint path provided. Using the default model.", file=sys.stderr)
            self.model = self.model_class(self.cfg)
        else:
            print(f"Loading model from checkpoint: {ckpt_path}", file=sys.stderr)
            # if not os.path.exists(ckpt_path):
            #     raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
            self.model = self.model_class.load_from_checkpoint(
                ckpt_path,
                map_location="cpu"
            )
        self.model.eval()
        self.model = self.model.to(self.accelerator.device)

        del self.model.ema_model
        self.model.ema_model = None

    def get_output(self, evaluated_points, *args, **kwargs):
        # print("evaluated_points norm", torch.norm(evaluated_points).detach(), file=sys.stderr)
        # print("self.model.dtype", next(self.model.parameters()).dtype, file=sys.stderr)
        predicted_output = self.model(evaluated_points)
        # print("predicted_output norm", torch.norm(predicted_output).detach(), file=sys.stderr)
        predicted_prob = torch.softmax(predicted_output, dim=-1)
        # print("predicted_prob norm", torch.norm(predicted_prob).detach(), file=sys.stderr)
        # predicted_entropy = -torch.sum(predicted_prob * torch.log(predicted_prob + 1e-10), dim=-1)
        predicted_entropy = -torch.sum(predicted_prob * torch.log(predicted_prob + 1e-8), dim=-1)
        return predicted_entropy

    def get_output_for_grad(self, 
                diffusion_model_output, diffusion_model_noisy_input, noise_level, alphas_cumprod, *args, **kwargs):
        """
        Generate guidance based on the input data.

        Args:
            model: The model for which guidance is to be generated.
            input_data: The input data for which guidance is to be generated.
        
        Returns:
            The generated guidance.
            (gradients, metrics)
        """
        # diffusion_output = kwargs.get("diffusion_model_output", None)
        diffusion_output = diffusion_model_output
        if diffusion_output is None:
            raise ValueError("diffusion_model_output is required to compute gradients.")

        scaled_input = self.scaler_input.get_scaled_input(
            noisy_input=diffusion_model_noisy_input,
            diffusion_output=diffusion_output,
            noise_level=noise_level,
            alphas_cumprod=alphas_cumprod,
        )

        # diffusion_output = diffusion_output.to(model.device)
        if self.cfg.generative_oracle.dataset_name != "cifar10":
            decoded_output = self.diffusers_pipe.vae.decode(scaled_input / self.diffusers_pipe.vae.config.scaling_factor, return_dict=False)[0]
            decoded_output = self.diffusers_pipe.image_processor.postprocess(
                image=decoded_output, output_type="pt", do_denormalize=[True] * len(decoded_output)).to(self.diffusers_pipe.vae.device)
            post_processed_scaled_input = self.post_process(decoded_output)
            scaled_input = post_processed_scaled_input
        # predict_output_input = (scaled_input, None) # None is a placeholder for the second input, which is not used in this case
        predict_output_input = scaled_input

        predicted_entropy = self.get_output(predict_output_input, *args, **kwargs)
        predicted_entropy_sum = predicted_entropy.mean()
        predicted_entropy_sum = -predicted_entropy_sum # For maximization

        # print("entropy entered!", file=sys.stderr)

        return predicted_entropy_sum, predicted_entropy

# class PredictiveEntropyGuidanceModel(GuidanceModel):
#     def __init__(self, cfg, accelerator, diffusers_pipe=None, 
#                  model_ckpt_path=None, train_dataloader=None, val_dataloader=None, datamodule=None,
#                  dummy_flag=False, *args, **kwargs):
#         from models.target_model_framework import TargetModelFramework
#         self.cfg = cfg
#         self.accelerator = accelerator
#         self.diffusers_pipe = diffusers_pipe
#         self.model_class = TargetModelFramework
#         self.scaler_input = ModelInputScaler(
#             time_dependent=False,
#             noise_scheduler_type=cfg.generative_oracle.noise_scheduler_type,
#             diffusion_model_type=cfg.generative_oracle.diffusion_model_type
#         )
#         if self.cfg.generative_oracle.dataset_name != "cifar10":
#             def post_process_fn(image):
#                 image = process_image(image)
#                 image = imagenet_normalization(image)
#                 return image
#             self.post_process = post_process_fn
#         self.load_model(model_ckpt_path)

#         assert (train_dataloader is not None and val_dataloader is not None) or datamodule is not None, \
#             "Either (train_dataloader and val_dataloader) or datamodule must be provided to precompute the cache."
        
#         if datamodule is not None:
#             # datamodule.setup()
#             self.datamodule = datamodule
#             self.train_dataloader = datamodule.train_dataloader()
#             self.val_dataloader = datamodule.holdout_dataloader()
#         else:
#             self.train_dataloader = train_dataloader
#             self.val_dataloader = val_dataloader

#         self.dummy_flag = dummy_flag
#         # if not dummy_flag: # This is for cycle 0 
#         #     self.precomputed_cache = self.precompute_cache(train_dataloader, val_dataloader)
#         self.precomputed_cache = None

#         self.current_holdout_set_class_idx = None
#         self.hidden_state_list = []
#         # self.precomputed_cache = self.precompute_cache(self.train_dataloader)

#         self.count = 0


#     def load_model(self, ckpt_path):
#         if ckpt_path is None:
#             print("No checkpoint path provided. Using the default model.", file=sys.stderr)
#             self.model = self.model_class(self.cfg)
#         else:
#             print(f"Loading model from checkpoint: {ckpt_path}", file=sys.stderr)
#             # if not os.path.exists(ckpt_path):
#             #     raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
#             self.model = self.model_class.load_from_checkpoint(
#                 ckpt_path,
#                 map_location="cpu"
#             )
#         self.model.eval()
#         self.model = self.model.to(self.accelerator.device)
#         self.set_hook_for_feature_extraction()
    
#     def forward_hook(self, module, input, output):
#         one_for_bias = torch.ones(input[0].shape[0], 1, device=input[0].device, dtype=input[0].dtype)
#         # self.hidden_state_list.append(input[0])
#         self.hidden_state_list.append(torch.cat([input[0], one_for_bias], dim=1)) # Adding bias term
    
#     def set_hook_for_feature_extraction(self):
#         if self.cfg.target_model.name in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]:
#             if self.cfg.train.ema.enabled:
#                 del self.model.ema_model
#                 self.model.ema_model = None
#                 gc.collect()
#                 torch.cuda.empty_cache()
#             # target_layer = self.model.target_model.model.model.linear if not self.cfg.train.ema.enabled else self.model.ema_model.model.linear
#             target_layer = self.model.target_model.model.model.linear
#             if not target_layer._forward_hooks:
#                 target_layer.register_forward_hook(self.forward_hook)
#             else:
#                 print("Hook already exists. Skipping hook registration.", file=sys.stderr)
#         elif self.cfg.target_model.name in ["vit_b", "vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32"]:
#             if self.cfg.train.ema.enabled:
#                 del self.model.ema_model
#                 self.model.ema_model = None
#                 gc.collect()
#                 torch.cuda.empty_cache()
#             target_layer = self.model.target_model.model.model.mlp_head
#             if not target_layer._forward_hooks:
#                 target_layer.register_forward_hook(self.forward_hook)
#             else:
#                 print("Hook already exists. Skipping hook registration.", file=sys.stderr)
#         else:
#             raise NotImplementedError(f"Feature extraction hook not implemented for model {self.cfg.target_model.name}")

#         # target_layer = self.model.target_model.model.model.linear
#         if target_layer._forward_hooks:
#             print(f"\n✅ Hook successfully registered to '{target_layer.__class__.__name__}'.")
#             print(f"Registered hooks: {target_layer._forward_hooks}\n")
#         else:
#             print(f"\n❌ Failed to register hook on '{target_layer.__class__.__name__}'.\n")
    
#     def precompute_cache(self, train_dataloader, *args, **kwargs):
#         # Assume that we ignore the interaction between different classes for now
#         hidden_states = []
#         logits_tensors = []

#         # tmp
#         with torch.autocast(device_type=self.accelerator.device.type, enabled=False):
#             train_dataloader = self.accelerator.prepare(train_dataloader)

#             with torch.no_grad():
#                 for train_data, train_label in tqdm(train_dataloader, desc="Precomputing cache for predictive entropy guidance model"):
#                     train_data = train_data.to(self.accelerator.device)
#                     train_label = train_label.to(self.accelerator.device)

#                     output = self.model(train_data)

#                     logits_tensors.append(output.detach().clone())
#                     hidden_states.append(torch.cat(self.hidden_state_list, dim=0).detach().clone())
#                     self.hidden_state_list.clear()

#                     if _USE_XLA:
#                         torch_xla.sync()
#             hidden_states = torch.cat(hidden_states, dim=0)
#             logits_tensors = torch.cat(logits_tensors, dim=0)

#             hidden_states = self.accelerator.gather(hidden_states) # N, feature_dim
#             logits_tensors = self.accelerator.gather(logits_tensors) # N, num_classes

#             hidden_states = hidden_states.to(torch.float64) ## TMP
#             logits_tensors = logits_tensors.to(torch.float64) ## TMP

#             # print("hidden_states nan:", torch.isnan(hidden_states).any().item(), file=sys.stderr)
#             # print("logits_tensors nan:", torch.isnan(logits_tensors).any().item(), file=sys.stderr)

#             softmax_probs = torch.softmax(logits_tensors, dim=-1) # N, num_classes
#             p_1_p = softmax_probs * (1 - softmax_probs) # N, num_classes


#             classwise_hessian = torch.zeros(p_1_p.size(1), hidden_states.size(1), hidden_states.size(1), device=hidden_states.device, dtype=torch.float64)
#             # torch einsum with buffer
#             buffer_size = 1024 * 8
#             for buffer_idx in range(0, p_1_p.size(0), buffer_size):
#                 end_idx = min(buffer_idx + buffer_size, p_1_p.size(0))
#                 p_1_p_buffer = p_1_p[buffer_idx:end_idx]
#                 hidden_states_buffer = hidden_states[buffer_idx:end_idx]
#                 classwise_hessian += torch.einsum(
#                     'nc,nj,nk->cjk',
#                     p_1_p_buffer,
#                     hidden_states_buffer,
#                     hidden_states_buffer,
#                 )

#             hessian_lambda = self.cfg.guidance_model.get("hessian_lambda", 0.02) * p_1_p.size(0) # Regularization term for Hessian
#             classwise_hessian = classwise_hessian + hessian_lambda * torch.eye(classwise_hessian.size(1), device=classwise_hessian.device, dtype=torch.float32).unsqueeze(0)

#             classwise_vs_cholesky = {}
#             for class_idx in range(classwise_hessian.size(0)):
#                 ith_classwise_hessian = classwise_hessian[class_idx]
#                 # choleskey = torch.cholesky(ith_classwise_hessian)
#                 ith_classwise_hessian = (ith_classwise_hessian + ith_classwise_hessian.t()) / 2.0 # Make it symmetric
#                 # print("first element", ith_classwise_hessian[0, 0], file=sys.stderr)
#                 choleskey = torch.linalg.cholesky(ith_classwise_hessian)
#                 classwise_vs_cholesky[class_idx] = choleskey
#             self.classwise_vs_cholesky = classwise_vs_cholesky

#     def get_output_for_grad(self, diffusion_model_output, diffusion_model_noisy_input, 
#                             diffusion_model_class_labels, noise_level, alphas_cumprod, *args, **kwargs):
#         diffusion_output = diffusion_model_output
#         if diffusion_output is None:
#             raise ValueError("diffusion_model_output is required to compute gradients.")
#         scaled_input = self.scaler_input.get_scaled_input(
#             noisy_input=diffusion_model_noisy_input,
#             diffusion_output=diffusion_output,
#             noise_level=noise_level,
#             alphas_cumprod=alphas_cumprod,
#         )

#         # diffusion_output = diffusion_output.to(model.device)
#         if self.cfg.generative_oracle.dataset_name != "cifar10":
#             # breakpoint()
#             decoded_output = self.diffusers_pipe.vae.decode(scaled_input / self.diffusers_pipe.vae.config.scaling_factor, return_dict=False)[0]
#             decoded_output = self.diffusers_pipe.image_processor.postprocess(
#                 image=decoded_output, output_type="pt", do_denormalize=[True] * len(decoded_output)).to(self.diffusers_pipe.vae.device)
#             post_processed_scaled_input = self.post_process(decoded_output)
#             scaled_input = post_processed_scaled_input
#         predictive_entropy = self.get_output(scaled_input, diffusion_model_class_labels)
#         if predictive_entropy is None:
#             return None, None
#         # aggregated_influence_score = influence_score.mean(dim=0)

#         # return aggregated_influence_score, influence_score
#         predicted_entropy_sum = predictive_entropy.mean()
#         predicted_entropy_sum = -predicted_entropy_sum # For maximization
#         return predicted_entropy_sum, predictive_entropy

#     def get_output(self, eval_point_inputs, eval_point_labels):
#         evaluated_inputs = eval_point_inputs.to(self.accelerator.device)
#         evaluated_labels = eval_point_labels.to(self.accelerator.device)

#         evaluated_output = self.model(evaluated_inputs) # B, num_classes
#         evaluated_features = torch.cat(self.hidden_state_list, dim=0) # B, feature_dim
#         self.hidden_state_list.clear()

        
class PredictiveEntropyGuidanceModel(GuidanceModel):
    def __init__(self, cfg, accelerator, diffusers_pipe=None, 
                 model_ckpt_path=None, train_dataloader=None, val_dataloader=None, datamodule=None,
                 dummy_flag=False, *args, **kwargs):
        from models.target_model_framework import TargetModelFramework
        self.cfg = cfg
        self.accelerator = accelerator
        self.diffusers_pipe = diffusers_pipe
        self.model_class = TargetModelFramework
        self.scaler_input = ModelInputScaler(
            time_dependent=False,
            noise_scheduler_type=cfg.generative_oracle.noise_scheduler_type,
            diffusion_model_type=cfg.generative_oracle.diffusion_model_type
        )
        if self.cfg.generative_oracle.dataset_name != "cifar10":
            def post_process_fn(image):
                image = process_image(image)
                image = imagenet_normalization(image)
                return image
            self.post_process = post_process_fn
        self.load_model(model_ckpt_path)

        assert (train_dataloader is not None and val_dataloader is not None) or datamodule is not None, \
            "Either (train_dataloader and val_dataloader) or datamodule must be provided to precompute the cache."
        
        if datamodule is not None:
            # datamodule.setup()
            self.datamodule = datamodule
            self.train_dataloader = datamodule.train_dataloader()
            self.val_dataloader = datamodule.holdout_dataloader()
        else:
            self.train_dataloader = train_dataloader
            self.val_dataloader = val_dataloader

        self.dummy_flag = dummy_flag
        # if not dummy_flag: # This is for cycle 0 
        #     self.precomputed_cache = self.precompute_cache(train_dataloader, val_dataloader)
        self.precomputed_cache = None

        self.current_holdout_set_class_idx = None
        self.hidden_state_list = []
        # self.precomputed_cache = self.precompute_cache(self.train_dataloader)

        self.count = 0
        self.average_output_of_train_dataset = None


    def load_model(self, ckpt_path):
        if ckpt_path is None:
            print("No checkpoint path provided. Using the default model.", file=sys.stderr)
            self.model = self.model_class(self.cfg)
        else:
            print(f"Loading model from checkpoint: {ckpt_path}", file=sys.stderr)
            # if not os.path.exists(ckpt_path):
            #     raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
            self.model = self.model_class.load_from_checkpoint(
                ckpt_path,
                map_location="cpu"
            )
        self.model.eval()
        self.model = self.model.to(self.accelerator.device)
        self.set_hook_for_feature_extraction()
    
    def forward_hook(self, module, input, output):
        one_for_bias = torch.ones(input[0].shape[0], 1, device=input[0].device, dtype=input[0].dtype)
        # self.hidden_state_list.append(input[0])
        self.hidden_state_list.append(torch.cat([input[0], one_for_bias], dim=1)) # Adding bias term
    
    def set_hook_for_feature_extraction(self):
        if self.cfg.target_model.name in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]:
            if self.cfg.train.ema.enabled:
                del self.model.ema_model
                self.model.ema_model = None
                gc.collect()
                torch.cuda.empty_cache()
            # target_layer = self.model.target_model.model.model.linear if not self.cfg.train.ema.enabled else self.model.ema_model.model.linear
            self.target_layer = self.model.target_model.model.model.linear
            if not self.target_layer._forward_hooks:
                self.target_layer.register_forward_hook(self.forward_hook)
            else:
                print("Hook already exists. Skipping hook registration.", file=sys.stderr)
        elif self.cfg.target_model.name in ["vit_b", "vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32"]:
            if self.cfg.train.ema.enabled:
                del self.model.ema_model
                self.model.ema_model = None
                gc.collect()
                torch.cuda.empty_cache()
            self.target_layer = self.model.target_model.model.model.mlp_head
            if not self.target_layer._forward_hooks:
                self.target_layer.register_forward_hook(self.forward_hook)
            else:
                print("Hook already exists. Skipping hook registration.", file=sys.stderr)
        else:
            raise NotImplementedError(f"Feature extraction hook not implemented for model {self.cfg.target_model.name}")

        # target_layer = self.model.target_model.model.model.linear
        if self.target_layer._forward_hooks:
            print(f"\n✅ Hook successfully registered to '{self.target_layer.__class__.__name__}'.")
            print(f"Registered hooks: {self.target_layer._forward_hooks}\n")
        else:
            print(f"\n❌ Failed to register hook on '{self.target_layer.__class__.__name__}'.\n")
    
    def get_hessian_lambda_by_optimization(self, classwise_hessian, initial_value=None):
        # get initialization of tau
        with torch.enable_grad():
            if initial_value is not None:
                print(f"Using initial value for tau optimization: {initial_value}", file=sys.stderr)
                log_tau = torch.log(torch.tensor(initial_value, device=classwise_hessian.device, dtype=classwise_hessian.dtype)).requires_grad_(True)
            else:
                log_tau = torch.randn(1, device=classwise_hessian.device, dtype=classwise_hessian.dtype).requires_grad_(True)

            # get the eigenvalues of the classwise_hessian
            eigenvalues = torch.linalg.eigvalsh(classwise_hessian)
            with torch.no_grad():
                l2_norm_last_layer_params = sum([p.pow(2).sum() for p in self.target_layer.parameters()])
                num_last_layer_params = sum(p.numel() for p in self.target_layer.parameters())
                # compute the l2 norm of the last layer parameters
            

            num_steps = 5000
            # optimizer = torch.optim.SGD([log_tau], lr=1e-5, momentum=0.9)
            optimizer = torch.optim.SGD([log_tau], lr=1e-3, momentum=0.9)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)

            def log_marginal_likelihood(eigenvalues, log_tau_var):
                tau = torch.exp(log_tau_var)
                # n = eigenvalues.size(0)
                log_det = torch.sum(torch.log(eigenvalues + tau))
                return -0.5 * log_det - 0.5 * tau * (l2_norm_last_layer_params) + 0.5 * num_last_layer_params * log_tau_var
                # return lml
            
            prev_loss = torch.finfo(torch.float32).max
            for step in range(num_steps):
                optimizer.zero_grad()
                lml = log_marginal_likelihood(eigenvalues, log_tau)
                loss = -lml
                loss.backward()
                optimizer.step()
                scheduler.step()
                if step % 500 == 0:
                    print(f"Step {step}: tau = {torch.exp(log_tau).item():.4f}, lml = {lml.item():.4f}", file=sys.stderr)
                
                if abs((prev_loss - loss.item()) / loss.item()) < 1e-6:
                    print(f"Converged at step {step}.", file=sys.stderr)
                    break
                else:
                    prev_loss = loss.item()
            return torch.exp(log_tau).item()

    
    def precompute_cache(self, train_dataloader, *args, **kwargs):
        # Assume that we ignore the interaction between different classes for now
        hidden_states = []
        logits_tensors = []

        train_dataloader = self.accelerator.prepare(train_dataloader)

        with torch.no_grad():
            for train_data, train_label in tqdm(train_dataloader, desc="Precomputing cache for predictive entropy guidance model"):
                train_data = train_data.to(self.accelerator.device)
                train_label = train_label.to(self.accelerator.device)

                output = self.model(train_data)

                logits_tensors.append(output.detach().clone())
                hidden_states.append(torch.cat(self.hidden_state_list, dim=0).detach().clone())
                self.hidden_state_list.clear()
        hidden_states = torch.cat(hidden_states, dim=0)
        logits_tensors = torch.cat(logits_tensors, dim=0)

        hidden_states = self.accelerator.gather(hidden_states) # N, feature_dim
        logits_tensors = self.accelerator.gather(logits_tensors) # N, num_classes

        hidden_states = hidden_states.to(torch.float64) ## TMP
        logits_tensors = logits_tensors.to(torch.float64) ## TMP

        # print("hidden_states nan:", torch.isnan(hidden_states).any().item(), file=sys.stderr)
        # print("logits_tensors nan:", torch.isnan(logits_tensors).any().item(), file=sys.stderr)

        softmax_probs = torch.softmax(logits_tensors, dim=-1) # N, num_classes
        p_1_p = softmax_probs * (1 - softmax_probs) # N, num_classes


        classwise_hessian = torch.zeros(p_1_p.size(1), hidden_states.size(1), hidden_states.size(1), device=hidden_states.device, dtype=torch.float64)
        # torch einsum with buffer
        buffer_size = 1024 * 8
        for buffer_idx in range(0, p_1_p.size(0), buffer_size):
            end_idx = min(buffer_idx + buffer_size, p_1_p.size(0))
            p_1_p_buffer = p_1_p[buffer_idx:end_idx]
            hidden_states_buffer = hidden_states[buffer_idx:end_idx]
            classwise_hessian += torch.einsum(
                'nc,nj,nk->cjk',
                p_1_p_buffer,
                hidden_states_buffer,
                hidden_states_buffer,
            )

        if self.cfg.guidance_model.get("tau_optimization", False):
            initial_value = self.cfg.guidance_model.get("hessian_lambda", 0.02) * p_1_p.size(0)
            hessian_lambda = self.get_hessian_lambda_by_optimization(classwise_hessian, initial_value=initial_value)
            print(f"Optimized hessian lambda: {hessian_lambda}", file=sys.stderr)
            hessian_lambda = self.accelerator.gather(torch.tensor(hessian_lambda, device=classwise_hessian.device, dtype=torch.float32))
            print(hessian_lambda, file=sys.stderr)
            hessian_lambda = torch.mean(hessian_lambda).item() 
            print(f"Mean optimized hessian lambda across all processes: {hessian_lambda}", file=sys.stderr)
        else:
            hessian_lambda = self.cfg.guidance_model.get("hessian_lambda", 0.02) * p_1_p.size(0) # Regularization term for Hessian
        classwise_hessian = classwise_hessian + hessian_lambda * torch.eye(classwise_hessian.size(1), device=classwise_hessian.device, dtype=torch.float32).unsqueeze(0)

        classwise_vs_cholesky = {}
        for class_idx in range(classwise_hessian.size(0)):
            ith_classwise_hessian = classwise_hessian[class_idx]
            # choleskey = torch.cholesky(ith_classwise_hessian)
            ith_classwise_hessian = (ith_classwise_hessian + ith_classwise_hessian.t()) / 2.0 # Make it symmetric
            choleskey = torch.linalg.cholesky(ith_classwise_hessian)
            classwise_vs_cholesky[class_idx] = choleskey
        self.classwise_vs_cholesky = classwise_vs_cholesky

    def get_output_for_grad(self, diffusion_model_output, diffusion_model_noisy_input, 
                            diffusion_model_class_labels, noise_level, alphas_cumprod, *args, **kwargs):
        diffusion_output = diffusion_model_output
        if diffusion_output is None:
            raise ValueError("diffusion_model_output is required to compute gradients.")
        scaled_input = self.scaler_input.get_scaled_input(
            noisy_input=diffusion_model_noisy_input,
            diffusion_output=diffusion_output,
            noise_level=noise_level,
            alphas_cumprod=alphas_cumprod,
        )

        # diffusion_output = diffusion_output.to(model.device)
        if self.cfg.generative_oracle.dataset_name != "cifar10":
            # breakpoint()
            decoded_output = self.diffusers_pipe.vae.decode(scaled_input / self.diffusers_pipe.vae.config.scaling_factor, return_dict=False)[0]
            decoded_output = self.diffusers_pipe.image_processor.postprocess(
                image=decoded_output, output_type="pt", do_denormalize=[True] * len(decoded_output)).to(self.diffusers_pipe.vae.device)
            post_processed_scaled_input = self.post_process(decoded_output)
            scaled_input = post_processed_scaled_input
        predictive_entropy = self.get_output(scaled_input, diffusion_model_class_labels)
        if predictive_entropy is None:
            return None, None
        # aggregated_influence_score = influence_score.mean(dim=0)

        # return aggregated_influence_score, influence_score
        predicted_entropy_sum = predictive_entropy.mean()
        predicted_entropy_sum = -predicted_entropy_sum # For maximization
        return predicted_entropy_sum, predictive_entropy

    def get_output(self, eval_point_inputs, eval_point_labels):
        evaluated_inputs = eval_point_inputs.to(self.accelerator.device)
        evaluated_labels = eval_point_labels.to(self.accelerator.device)

        evaluated_output = self.model(evaluated_inputs) # B, num_classes
        evaluated_features = torch.cat(self.hidden_state_list, dim=0) # B, feature_dim
        self.hidden_state_list.clear()

        # print("evaluated_output: ", evaluated_output, file=sys.stderr)
        

        # if self.accelerator.is_main_process:
        #     # torch.save(eval_point_inputs.cpu(), f"z_bald_predictive_entropy_tmp/eval_point_inputs_{self.count}.pt")
        #     # torch.save(eval_point_labels.cpu(), f"z_bald_predictive_entropy_tmp/eval_point_labels_{self.count}.pt")
        #     torch.save(evaluated_output.cpu(), f"z_bald_predictive_entropy_tmp/evaluated_output_{self.count}.pt")
        #     torch.save(evaluated_features.cpu(), f"z_bald_predictive_entropy_tmp/evaluated_features_{self.count}.pt")
        # self.count += 1

        # if 
        if not hasattr(self, "classwise_vs_cholesky"):
            self.precompute_cache(self.train_dataloader)

        classwise_vc_tensor = torch.zeros_like(evaluated_output, device=evaluated_features.device)
        for i in self.classwise_vs_cholesky:
            # classwise_vc = torch.cholesky_solve(evaluated_features.t(), self.classwise_vs_cholesky[i])
            classwise_vc = torch.cholesky_solve(evaluated_features.t().to(torch.float32), self.classwise_vs_cholesky[i].to(torch.float32))
            classwise_vc = classwise_vc.t()
            classwise_vc_diag = torch.sum(evaluated_features * classwise_vc, dim=-1) # B,
            classwise_vc_tensor[:, i] = classwise_vc_diag
        # breakpoint()
        scaled_output = evaluated_output / torch.sqrt(1 + torch.pi / 8 * classwise_vc_tensor)
        scaled_prob = torch.softmax(scaled_output, dim=-1)
        predictive_entropy = - torch.sum(scaled_prob * torch.log(scaled_prob + 1e-8), dim=-1)
        return predictive_entropy
    
    def get_average_output_of_train_dataset(self):
        if self.average_output_of_train_dataset is not None:
            return self.average_output_of_train_dataset
        
        all_outputs = []
        self.model.eval()
        with torch.no_grad():
            for train_data, train_label in tqdm(self.train_dataloader, desc="Computing average output of train dataset"):
                train_data = train_data.to(self.accelerator.device)
                train_label = train_label.to(self.accelerator.device)
                output = self.get_output(train_data, train_label)
                all_outputs.append(output.detach().cpu())
        all_outputs = torch.cat(all_outputs, dim=0)
        self.average_output_of_train_dataset = torch.mean(all_outputs, dim=0)
        return self.average_output_of_train_dataset

        

    
    

class BALDGuidanceModel(PredictiveEntropyGuidanceModel):
    def __init__(self, cfg, accelerator, diffusers_pipe=None, 
                 model_ckpt_path=None, train_dataloader=None, val_dataloader=None, datamodule=None,
                 dummy_flag=False, *args, **kwargs):
        super().__init__(cfg, accelerator, diffusers_pipe, model_ckpt_path, train_dataloader, val_dataloader, datamodule, dummy_flag, *args, **kwargs)

    def get_output(self, eval_point_inputs, eval_point_labels):
        evaluated_inputs = eval_point_inputs.to(self.accelerator.device)
        # evaluated_labels = eval_point_labels.to(self.accelerator.device)

        evaluated_output = self.model(evaluated_inputs) # B, num_classes
        evaluated_features = torch.cat(self.hidden_state_list, dim=0) # B, feature_dim
        self.hidden_state_list.clear()

        if not hasattr(self, "classwise_vs_cholesky"):
            self.precompute_cache(self.train_dataloader)

        classwise_vc_tensor = torch.zeros_like(evaluated_output, device=evaluated_features.device) # B, num_classes
        for i in self.classwise_vs_cholesky:
            classwise_vc = torch.cholesky_solve(evaluated_features.t().to(torch.float32), self.classwise_vs_cholesky[i].to(torch.float32))
            classwise_vc = classwise_vc.t()
            classwise_vc_diag = torch.sum(evaluated_features * classwise_vc, dim=-1) # B,
            classwise_vc_tensor[:, i] = classwise_vc_diag
        scaled_output = evaluated_output / torch.sqrt(1 + torch.pi / 8 * classwise_vc_tensor)
        scaled_prob = torch.softmax(scaled_output, dim=-1)

        classwise_vc_tensor_tilde = classwise_vc_tensor / (1 + torch.pi / 8 * classwise_vc_tensor)
        bald = 0.5 * torch.sum(scaled_prob * (1 - scaled_prob) * classwise_vc_tensor_tilde, dim=-1)
        return bald

class NoGuidanceModel(GuidanceModel):
    """
    A dummy guidance model that does not provide any guidance.
    This can be used when no guidance is needed.
    """
    def get_output_for_grad(self, *args, **kwargs):
        return None, None

    def get_output(self, *args, **kwargs):
        return None

    def load_model(self, *args, **kwargs):
        pass
