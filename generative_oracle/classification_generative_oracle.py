import os
import sys
import yaml

import torch
import lightning as L
import numpy as np

from .base_generative_oracle import GuidanceBasedGenerativeOracle
from .guidance_model_container import GuidanceModelContainer
from .stable_diffusion.custom_stable_diffusion import CustomStableDiffusionPipeline
from .FKD.fkd_pipeline_sd import FKDStableDiffusion

from utils.data_utils import get_class_idx_to_class_name_mapping
# from .utils.data_utils import get_class_idx_to_class_name_mapping

from functools import partial

import random

from diffusers import DDIMScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler

try:
    import torch_xla
    _USE_XLA = True
except ImportError:
    _USE_XLA = False


class ClassificationGenerativeOracle(GuidanceBasedGenerativeOracle):
    """
    A generative oracle for classification tasks that uses guidance-based methods to generate samples.
    Generative Oracle Utils consists of two main components:
        1. Generative Oracle: A generative oracle for generating samples based on the target
        2. Guidance Model: A model that provides guidance for the generative oracle.
    """
    # def __init__(self, cfg, trainer):
    def __init__(self, cfg, accelerator, 
                 *args, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.accelerator = accelerator
        # self.xm =  None
        self.torch_xla = None
        self.necessary_components = {
            "generate": [
                "inference_steps",
                "guidance_scale",
                "labels",
                "num_samples_per_label"
            ]
        }

        self.args = args
        self.kwargs = kwargs
        
        self.create_generative_oracle(cfg)
        self.create_guidance_model(cfg)
        self.setup()

    def setup(self):
        if "xla" in str(self.accelerator.device):
            import torch_xla
            self.torch_xla = torch_xla
        self.generative_model = self.generative_model.to(self.accelerator.device)

    def get_to_device(self):
        return self.accelerator.device

    def create_generative_oracle(self, cfg):
        super().create_generative_oracle(cfg)
        
        if cfg.dataset.name in ["imagenet_100", "imagenette"]:
            torch_model_dtype = torch.float32
            # if self.cfg.train.mixed_precision:
            #     torch_model_dtype = torch.bfloat16
            if self.cfg.generative_oracle.get("mixed_precision", self.cfg.train.get("mixed_precision", False)):
                torch_model_dtype = torch.bfloat16

            print("Using torch model dtype:", torch_model_dtype, file=sys.stderr)
            

            cfg.generative_oracle.guidance_type = cfg.generative_oracle.get("guidance_type", "gradient")
            if cfg.generative_oracle.guidance_type == "gradient":
                self.generative_model = CustomStableDiffusionPipeline.from_pretrained(
                    self.cfg.generative_oracle.pretrained_model,
                    torch_dtype=torch_model_dtype,
                    safety_checker=None
                )
                self.generative_model.scheduler = DDIMScheduler.from_config(
                    self.generative_model.scheduler.config
                )
                
            elif cfg.generative_oracle.guidance_type == "fkd":
                # pass
                self.generative_model = FKDStableDiffusion.from_pretrained(
                    self.cfg.generative_oracle.pretrained_model,
                    torch_dtype=torch_model_dtype,
                    safety_checker=None
                )
                # self.generative_model.scheduler = EulerAncestralDiscreteScheduler.from_config(
                #     self.generative_model.scheduler.config
                # )
                self.generative_model.scheduler = DDIMScheduler.from_config(
                    self.generative_model.scheduler.config
                )

            if self.cfg.generative_oracle.get("scheduler", None) is not None:
                if self.cfg.generative_oracle.scheduler.name == "euler":
                    self.generative_model.scheduler = EulerAncestralDiscreteScheduler.from_config(
                        self.generative_model.scheduler.config
                    )
                elif self.cfg.generative_oracle.scheduler.name == "ddim":
                    self.generative_model.scheduler = DDIMScheduler.from_config(
                        self.generative_model.scheduler.config
                    )
                # import sys
                # print("Scheduler set to:", self.cfg.generative_oracle.scheduler.name, file=sys.stderr)
                # breakpoint()

                # self.generative_model.scheduler.eta = 1.0
            # self.generative_model.vae.to(torch.float32)
            # self.generative_model = self.generative_model.to(device)

            self.generative_model.to(self.accelerator.device)
            
            self.freedom_callback_on_step_end_tensor_input = ["latents", "noise_pred", "prev_latents"]

            self.class_idx_to_label_map = get_class_idx_to_class_name_mapping(cfg)
            self.class_idx_to_label_map = {k: ", ".join(v) for k, v in self.class_idx_to_label_map.items()}

            class_idx_to_label_map_embed = {}
            with torch.no_grad():
                for class_idx in self.class_idx_to_label_map:
                    # print(f"Class idx {class_idx}: {self.class_idx_to_label_map[class_idx]}", file=sys.stderr)
                    prompt_embeds, _ = self.generative_model.encode_prompt(
                        prompt=self.class_idx_to_label_map[class_idx],
                        device=self.accelerator.device,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=False,
                        lora_scale=None,
                    )
                    class_idx_to_label_map_embed[class_idx] = prompt_embeds

                    if _USE_XLA:
                        torch_xla.sync()

            self.class_idx_to_label_map_embed = class_idx_to_label_map_embed

            tmp_diff_value_list = []
            def pre_freedom_iteration(diffusion_pipeline, step, timestep, 
                                        callback_kwargs, freedom_model, freedom_scale, 
                                        guide_start=0, guide_end=100):
                # step: increasing: 1 -> 100
                # timestep: decreasing: 1000 -> 1
                guidance_result = freedom_model.guidance(diffusion_pipeline, callback_kwargs, step, timestep, freedom_scale)
                tmp_diff_value_list.append(guidance_result["differentiate_value"])
                return guidance_result
            self.freedom_iteration = partial(pre_freedom_iteration)

        else:
            raise ValueError(f"Unsupported dataset {cfg.dataset.name} for classification generative oracle.")

    def create_guidance_model(self, cfg):
        self.guidance_model_container = GuidanceModelContainer(
            cfg=cfg, accelerator=self.accelerator,
            diffusers_pipe=self.generative_model,
            *self.args,
            **self.kwargs
        )
    
    def load_guidance_model(self, guidance_model_ckpt_path=None):
        # self.guidance_model_container.load_guidance_model(guidance_model_ckpt_path, self.accelerator.device)
        self.guidance_model_container.load_guidance_model(guidance_model_ckpt_path)

    def generate(self, *args, **kwargs):
        """
        Generate samples for classification tasks using the guidance-based method.
        """

        # device = self.get_to_device()
        # device = kwargs.get('device', self.get_to_device())

        device = self.accelerator.device
        # self.generative_model = self.generative_model.to(device)
        # Implement the generation logic specific to classification tasks here
        necessary_components = self.necessary_components['generate']
        for component in necessary_components:
            if component not in kwargs:
                raise ValueError(f"Missing necessary component for generation: {component}")
            
        inference_steps = kwargs['inference_steps']
        # guidance_scale = kwargs['guidance_scale']
        guidance_scale = kwargs['guidance_scale']
        labels = kwargs['labels']
        num_samples_per_label = kwargs['num_samples_per_label']
        seed = kwargs.get('seed', None)

        batch_size = num_samples_per_label * len(labels)
        # device = self.generative_model.device

        if self.cfg.dataset.name in ["imagenet_100", "imagenette"]:
            # For ImageNet-100, we use the DiffusionPipeline to generate images
            # raise NotImplementedError("ImageNet-100 generation is not implemented yet.")

            freedom_guidance_scale = guidance_scale
            # prompts = []
            prompt_embeds = []
            for i, label_idx in enumerate(labels):
                label_idx = label_idx.item()
                for _ in range(num_samples_per_label):
                    # prompts.append(self.class_idx_to_label_map[label_idx])
                    prompt_embeds.append(self.class_idx_to_label_map_embed[label_idx])
            prompt_embeds = torch.cat(prompt_embeds, dim=0)

            if "cuda" in str(self.accelerator.device):
                generator_device = self.accelerator.device
            else:
                generator_device = "cpu"
            generator = torch.Generator(device=generator_device).manual_seed(seed)
            input_dict = {
                "prompt_embeds": prompt_embeds,
                "class_labels": labels,
                "num_inference_steps": self.cfg.generative_oracle.inference_steps, 
                "callback_on_step_end_tensor_inputs": self.freedom_callback_on_step_end_tensor_input,
                "callback": lambda *args: self.torch_xla.sync() if "xla" in str(self.accelerator.device) else None, 
                "callback_steps": 1,
                "guidance_scale": self.cfg.generative_oracle.guidance_scale, # CFG, need to be fixed
                "freedom_guidance_scale": freedom_guidance_scale,
                "bbox_binary_mask": None,
                "guidance_model_container": self.guidance_model_container,
                "guidance_model_type": self.cfg.guidance_model.guidance_model_type,
                "generator": generator,
                "fkd_args": self.cfg.generative_oracle.fkd_args if self.cfg.generative_oracle.guidance_type == "fkd" else None,
                "accelerator": self.accelerator,
                "eta": 1.0 if self.cfg.generative_oracle.guidance_type == "fkd" else 0.0,
            }

            images = self.generative_model(**input_dict).images # Add aux
            aux = None

        else:
            raise ValueError(f"Unsupported dataset {self.cfg.dataset.name} for classification generative oracle.")

        return images, aux
