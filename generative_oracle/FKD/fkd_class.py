"""
Feynman-Kac Diffusion (FKD) steering mechanism implementation.
"""
import os
import sys
import yaml

import torch
from enum import Enum
import numpy as np
from typing import Callable, Optional, Tuple
import logging

from diffusers import DDIMScheduler, EulerAncestralDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor

class PotentialType(Enum):
    DIFF = "diff"
    MAX = "max"
    ADD = "add"
    RT = "rt"


class FKD:
    """
    Implements the FKD steering mechanism. Should be initialized along the diffusion process. .resample() should be invoked at each diffusion timestep.
    See FKD fkd_pipeline_sdxl
    Args:
        potential_type: Type of potential function must be one of PotentialType.
        lmbda: Lambda hyperparameter controlling weight scaling.
        num_particles: Number of particles to maintain in the population.
        adaptive_resampling: Whether to perform adaptive resampling.
        resample_frequency: Frequency (in timesteps) to perform resampling.
        resampling_t_start: Timestep to start resampling.
        resampling_t_end: Timestep to stop resampling.
        time_steps: Total number of timesteps in the sampling process.
        reward_fn: Function to compute rewards from decoded latents.
        reward_min_value: Minimum value for rewards (default: 0.0). Important for the Max potential type.
        latent_to_decode_fn: Function to decode latents to images, relevant for latent diffusion models (default: identity function).
        device: Device on which computations will be performed (default: CUDA).
        **kwargs: Additional keyword arguments, unused.
    """

    def __init__(
        self,
        *,
        potential_type: PotentialType,
        lmbda: float,
        num_particles: int,
        adaptive_resampling: bool,
        resample_frequency: int,
        resampling_t_start: int,
        resampling_t_end: int,
        time_steps: int,
        # reward_fn: Callable[[torch.Tensor], torch.Tensor],
        guidance_model_container, 
        # reward_min_value: float = 0.0,
        reward_min_value: float = torch.finfo(torch.float32).min,
        device: torch.device = torch.device('cuda'),
        **kwargs,
    ) -> None:
        # Initialize hyperparameters and functions

        # if kwargs:
            # logging.warning(f"FKD Steering - Unused arguments: {kwargs}")

        self.potential_type = PotentialType(potential_type)
        self.lmbda = lmbda
        self.num_particles = num_particles
        self.adaptive_resampling = adaptive_resampling
        self.resample_frequency = resample_frequency
        self.resampling_t_start = resampling_t_start
        self.resampling_t_end = resampling_t_end
        self.time_steps = time_steps

        # self.reward_fn = reward_fn
        self.guidance_model_container = guidance_model_container
        self.reward_fn = guidance_model_container.get_output_of_guidance_model
        # self.latent_to_decode_fn = latent_to_decode_fn

        # Initialize device and population reward state
        self.device = device

        # initial rewards
        self.reward_min_value = kwargs.get("reward_min_value", reward_min_value)
        print("self.reward_min_value:", self.reward_min_value)

        # self.population_rs = (
        #     torch.ones(self.num_particles, device=self.device) * reward_min_value
        # )
        self.population_rs = (
            torch.ones(self.num_particles, device=self.device) * self.reward_min_value
        )
        # self.population_rs = torch.ones(self.num_particles, device=self.device)
        self.product_of_potentials = torch.ones(self.num_particles).to(self.device)

        # self.cfg = kwargs.get("cfg", None)
        self.guidance_model_type = kwargs.get("guidance_model_type", None)
        self.proposal_distribution = kwargs.get("proposal_distribution", None)
        self.gradient_guidance = kwargs.get("gradient_guidance", 0.0)

        self.no_reweight = kwargs.get("no_reweight", False)
        self.reweight_except_last = kwargs.get("reweight_except_last", False)
        # self.kwargs = kwargs
        self.max_grad_reweight = kwargs.get("max_grad_reweight", False)
        self.adaptive_lmbda = kwargs.get("adaptive_lmbda", None)
        self.adaptive_lmbda_value = None

        # TMP
        self.debug_mode = kwargs.get("debug_mode", False)
        self.save_image_dir = kwargs.get("save_image_dir", None)
        self.process_idx = kwargs.get("process_idx", None)
        self.no_guidance = kwargs.get("no_guidance", False)



    def get_scale_factor_and_dt(self, scheduler, t_idx):
        if isinstance(scheduler, DDIMScheduler):
            # DDIM은 x_t = sqrt(alpha)*x0 + sqrt(1-alpha)*eps 구조
            # mu의 변화량과 eps의 변화량 관계는 sqrt(1-alpha)와 관련이 있으나, 
            # 단순화를 위해 1.0으로 두거나 ddim 전용 dt를 정의할 수 있습니다.
            return 1.0, -1.0 
        elif isinstance(scheduler, EulerAncestralDiscreteScheduler):
            t_idx = scheduler.index_for_timestep(t_idx, scheduler.timesteps)
            sigma_t = scheduler.sigmas[t_idx]
            # 인덱스 버그 수정: (t_idx + 1) < len(sigmas) 조건 확인
            sigma_t_1 = scheduler.sigmas[t_idx + 1] if (t_idx + 1) < len(scheduler.sigmas) else torch.tensor(0.0)

            sigma_up = torch.sqrt(sigma_t_1**2 * (sigma_t**2 - sigma_t_1**2) / sigma_t**2)
            sigma_down = torch.sqrt(sigma_t_1**2 - sigma_up**2)

            scale_factor = 1.0 / ((sigma_t**2 + 1) ** 0.5)
            dt = sigma_down - sigma_t
            return scale_factor, dt
        return 1.0, -1.0

    def get_sigma_up_and_sigma_down(self, scheduler, t_idx):
        if isinstance(scheduler, EulerAncestralDiscreteScheduler):
            t_idx = scheduler.index_for_timestep(t_idx, scheduler.timesteps)
            sigma_t = scheduler.sigmas[t_idx]
            # 인덱스 버그 수정: (t_idx + 1) < len(sigmas) 조건 확인
            sigma_t_1 = scheduler.sigmas[t_idx + 1] if (t_idx + 1) < len(scheduler.sigmas) else torch.tensor(0.0)

            sigma_up = torch.sqrt(sigma_t_1**2 * (sigma_t**2 - sigma_t_1**2) / sigma_t**2)
            sigma_down = torch.sqrt(sigma_t_1**2 - sigma_up**2)
            return sigma_up, sigma_down
        else:
            # raise NotImplementedError(f"scheduler type {type(scheduler)} not implemented in FKD gradient proposal.")
            return 1, 1

    def get_sigma_and_mu_theta(self, scheduler, t_idx, latents, noise_pred, prev_latents, eta):
        if isinstance(scheduler, DDIMScheduler):
            t_idx = int(t_idx)
            alpha_prod_t = scheduler.alphas_cumprod[t_idx]
            if t_idx > 0:
                alpha_prod_t_prev = scheduler.alphas_cumprod[t_idx-1]
            elif hasattr(scheduler, "final_alpha_cumprod"):
                alpha_prod_t_prev = scheduler.final_alpha_cumprod
            else:
                alpha_prod_t_prev = scheduler.alphas_cumprod[t_idx] 

            # 1. 분산(sigma_squared) 계산 (eta=1 기준)
            var_t = ((1 - alpha_prod_t_prev) / (1 - alpha_prod_t)) * (1 - alpha_prod_t / alpha_prod_t_prev)
            sigma_squared = eta**2 * var_t

            # 2. mu_theta 계산 (Original model mean)
            # pred_epsilon은 모델이 예측한 노이즈
            pred_x0 = (prev_latents - (1 - alpha_prod_t)**0.5 * noise_pred) / alpha_prod_t**0.5
            mu_theta = (alpha_prod_t_prev**0.5 * pred_x0 + (1 - alpha_prod_t_prev - sigma_squared)**0.5 * noise_pred)
        elif isinstance(scheduler, EulerAncestralDiscreteScheduler):
            t_idx = scheduler.index_for_timestep(t_idx, scheduler.timesteps)

            # 다음 시그마가 존재하면 가져오고, 마지막 단계면 0으로 설정
            sigma_t = scheduler.sigmas[t_idx]
            sigma_t_1 = scheduler.sigmas[t_idx + 1] if (t_idx + 1) < len(scheduler.sigmas) else torch.tensor(0.0, device=self.device)

            sigma_up = torch.sqrt((sigma_t_1**2) * (sigma_t**2 - sigma_t_1**2) / (sigma_t**2))
            sigma_down = torch.sqrt((sigma_t_1**2) - (sigma_up**2))

            pred_x0 = prev_latents - sigma_t * noise_pred
            mu_theta = prev_latents + (sigma_down  - sigma_t) / sigma_t * (prev_latents - pred_x0)
            sigma_squared = sigma_up**2
        else:
            raise NotImplementedError(f"scheduler type {type(scheduler)} not implemented in FKD gradient proposal.")
        return sigma_squared, mu_theta

    def get_unguided_prev_latents(self, scheduler, t_idx, latents, noise_pred, eta, generator):
        if isinstance(scheduler, EulerAncestralDiscreteScheduler):
            t_idx = scheduler.index_for_timestep(t_idx, scheduler.timesteps)

            # 다음 시그마가 존재하면 가져오고, 마지막 단계면 0으로 설정
            sigma_t = scheduler.sigmas[t_idx]
            sigma_t_1 = scheduler.sigmas[t_idx + 1] if (t_idx + 1) < len(scheduler.sigmas) else torch.tensor(0.0, device=self.device)

            sigma_up = torch.sqrt(sigma_t_1**2 * (sigma_t**2 - sigma_t_1**2) / sigma_t**2)
            sigma_down = torch.sqrt(sigma_t_1**2 - sigma_up**2)

            pred_x0 = latents - sigma_t * noise_pred
            derivative = (latents - pred_x0) / sigma_t
            dt = sigma_down - sigma_t
            prev_samples = latents + derivative * dt
            noise = randn_tensor(latents.shape, generator=generator, device=latents.device,dtype=latents.dtype)
            prev_latents = prev_samples + noise * sigma_up
        else:
            raise NotImplementedError(f"scheduler type {type(scheduler)} not implemented in FKD gradient proposal.")
        return prev_latents
            

    def resample(
        self, *, sampling_idx: int, latents: torch.Tensor,
        reward_model_input: dict, 
        noise_pred, new_noise_pred, 
        prev_latents, scheduler, t_idx, eta, generator) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Perform resampling of particles if conditions are met.
        Should be invoked at each timestep in the reverse diffusion process.

        Args:
            sampling_idx: Current sampling index (timestep).
            latents: Current noisy latents.
            x0_preds: Predictions for x0 based on latents.

        Returns:
            A tuple containing resampled latents and optionally resampled images.
        """
        # Check if resampling is within the allowed range and conditions
        resampling_interval = np.arange(
            self.resampling_t_start, self.resampling_t_end + 1, self.resample_frequency
        )
        resampling_interval = np.append(resampling_interval, self.time_steps - 1)

        # if sampling_idx not in resampling_interval:
        #     return latents
        if self.guidance_model_type == "no_guidance" or sampling_idx not in resampling_interval:
            return latents

        # Decode latents to population images and compute rewards
        # population_images = self.latent_to_decode_fn(x0_preds)
        # rs_candidates = self.reward_fn(population_images)
        _, rs_candidates = self.reward_fn(**reward_model_input)

        # if "influence_score" in self.guidance_model_type:
        # rs_candidates = -rs_candidates # For maximization. Assumes reward_fn returns cost, which should be minimize.
        
        # Compute importance weights
        if self.potential_type == PotentialType.MAX:
            is_candidate_updated = rs_candidates > self.population_rs
            rs_candidates = torch.max(rs_candidates, self.population_rs)
            w = torch.exp(self.lmbda * rs_candidates)
        elif self.potential_type == PotentialType.ADD:
            rs_candidates = rs_candidates + self.population_rs
            w = torch.exp(self.lmbda * rs_candidates)
        elif self.potential_type == PotentialType.DIFF:
            diffs = rs_candidates - self.population_rs
            w = torch.exp(self.lmbda * diffs)
        elif self.potential_type == PotentialType.RT:
            w = torch.exp(self.lmbda * rs_candidates)
        else:
            raise ValueError(f"potential_type {self.potential_type} not recognized")


        if sampling_idx == self.time_steps - 1:
            if (
                self.potential_type == PotentialType.MAX
                or self.potential_type == PotentialType.ADD
                or self.potential_type == PotentialType.RT
            ):
                w = torch.exp(self.lmbda * rs_candidates) / self.product_of_potentials

        # w = torch.clamp(w, 0, 1e10)
        # w[torch.isnan(w)] = 0.0
        raw_potential = w.clone()

        if self.proposal_distribution == "gradient" and not self.no_reweight \
            and not (self.reweight_except_last and sampling_idx == self.time_steps - 1):
            # t 단계에서 t-1로 가는 과정
            sigma_squared, mu_theta = self.get_sigma_and_mu_theta(
                scheduler=scheduler, t_idx=t_idx, latents=latents, noise_pred=noise_pred, prev_latents=prev_latents, eta=eta
            )
            # sigma_square, mu_tau_for_compare = self.get_sigma_and_mu_theta(
            #     scheduler=scheduler, t_idx=t_idx, latents=latents, noise_pred=new_noise_pred, prev_latents=prev_latents, eta=eta
            # )
            # sigma_squared, mu_tau = self.get_sigma_and_mu_theta(
            #     scheduler=scheduler, t_idx=t_idx, latents=latents, noise_pred=new_noise_pred, prev_latents=prev_latents, eta=eta
            # )
            mu_tau = mu_theta + sigma_squared * (new_noise_pred - noise_pred)
            
            # _, dt = self.get_scale_factor_and_dt(scheduler, t_idx)
            # scale_factor, dt = self.get_scale_factor_and_dt(scheduler, t_idx)
            # dt = -dt  # reverse direction, cause the grad is already computed to maximize the reward

            log_p_theta_over_tau = torch.sum((latents - mu_tau) ** 2, dim=(1, 2, 3)) - torch.sum((latents - mu_theta) ** 2, dim=(1, 2, 3))
            log_p_theta_over_tau = log_p_theta_over_tau / (2 * sigma_squared + 1e-7)
            p_theta_over_tau = torch.exp(log_p_theta_over_tau)

            # breakpoint()
            # if self.potential_type == PotentialType.MAX and self.cfg.generative_oracle.get("max_grad_reweight", False):
            # if self.potential_type == PotentialType.MAX and self.max_grad_reweight:
            #     p_theta_over_tau = torch.where(is_candidate_updated, p_theta_over_tau, torch.ones_like(p_theta_over_tau))
            #     unguided_prev_latents = self.get_unguided_prev_latents(
            #         scheduler=scheduler, t_idx=t_idx, latents=prev_latents, noise_pred=noise_pred, eta=eta, generator=generator
            #     )
            #     latents = torch.where(is_candidate_updated[:, None, None, None], prev_latents, unguided_prev_latents)
            w = w * p_theta_over_tau
        
        if self.adaptive_lmbda:
            if self.adaptive_lmbda_value is None:
                self.adaptive_lmbda_value = self.guidance_model_container.guidance_model.get_average_output_of_train_dataset() # TODO: ugly code. fix later
                print("Initial adaptive_lmbda_value:", self.adaptive_lmbda_value, file=sys.stderr)
                # self.adaptive_lmbda_value = 1.0 / (self.adaptive_lmbda_value + 1e-7)
                self.adaptive_lmbda_value = 0.5 / (self.adaptive_lmbda_value + 1e-7)
                print("adaptive_lmbda_value:", self.adaptive_lmbda_value, file=sys.stderr)
            # w = w * torch.exp(self.adaptive_lmbda_value)
            w = torch.exp(self.adaptive_lmbda_value * torch.log(w + 1e-7))

        w = torch.clamp(w, 0, 1e10)
        w[torch.isnan(w)] = 0.0

        if self.adaptive_resampling or sampling_idx == self.time_steps - 1:
            # compute effective sample size
            normalized_w = w / w.sum()
            ess = 1.0 / (normalized_w.pow(2).sum())

            if ess < 0.5 * self.num_particles:
                print(f"Resampling at timestep {sampling_idx} with ESS: {ess}", file=sys.stderr)
                # Resample indices based on weights
                indices = torch.multinomial(
                    w, num_samples=self.num_particles, replacement=True
                )
                ## TMP
                if self.no_guidance: 
                    indices = torch.arange(self.num_particles)
                resampled_latents = latents[indices]
                self.population_rs = rs_candidates[indices]

                # Resample population images
                # resampled_images = population_images[indices]

                # Update product of potentials; used for max and add potentials
                self.product_of_potentials = (
                    self.product_of_potentials[indices] * w[indices]
                )
            else:
                # No resampling
                # resampled_images = population_images
                indices = torch.arange(self.num_particles)
                resampled_latents = latents
                self.population_rs = rs_candidates

        else:
            # Resample indices based on weights
            indices = torch.multinomial(
                w, num_samples=self.num_particles, replacement=True
            )
            resampled_latents = latents[indices]
            self.population_rs = rs_candidates[indices]

            # Resample population images
            # resampled_images = population_images[indices]

            # Update product of potentials; used for max and add potentials
            self.product_of_potentials = (
                self.product_of_potentials[indices] * w[indices]
            )
        
        if self.debug_mode:
            # breakpoint()
            if not self.proposal_distribution == "gradient" or self.no_reweight:
                # log_p_theta_over_tau = torch.ones_like(w)
                log_p_theta_over_tau = torch.zeros_like(w)

            with open(os.path.join(self.save_image_dir, f"logging.yaml"), "a") as f:
                f.write("---\n")
                yaml.dump({
                    "sampling_idx": sampling_idx,
                    "log_p_theta_over_tau": log_p_theta_over_tau.cpu().tolist(),
                    "raw_potential": raw_potential.cpu().tolist(),
                    "potential": w.cpu().tolist(),
                    "reward": rs_candidates.cpu().tolist(),
                    "selected_indices": indices.cpu().tolist(),
                }, f, default_flow_style=False)
        return resampled_latents

    def propose(
        self, *, sampling_idx: int, t_idx, scheduler, pred_noise: torch.Tensor, reward_model_input: dict
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Placeholder for propose method.
        FKD does not require a separate propose step.
        """
        # Check if resampling is within the allowed range and conditions
        resampling_interval = np.arange(
            self.resampling_t_start, self.resampling_t_end + 1, self.resample_frequency
        )
        resampling_interval = np.append(resampling_interval, self.time_steps - 1)
        
        if self.guidance_model_type == "no_guidance" or \
            self.proposal_distribution is None or \
            sampling_idx not in resampling_interval:
            return pred_noise
        
        if self.proposal_distribution == "gradient":
            grad, aux_output = self.guidance_model_container.get_grad_of_guidance_model(**reward_model_input)
            scale_factor, dt = self.get_scale_factor_and_dt(scheduler, t_idx)
            # scaled_grad = grad * scale_factor / (dt + 1e-7)
            # dt is negative in reverse process, so negate the grad to ascend the reward
            # if self.grad_wrt_potential:
            #     grad = grad * self.gradient_guidance
            if self.no_reweight:
                pred_noise = pred_noise + self.gradient_guidance * grad
                return pred_noise
            else:
                # scaled_grad = (-grad) * scale_factor / dt
                # if isinstance(scheduler, EulerAncestralDiscreteScheduler):
                #     sigma_up, _ = self.get_sigma_up_and_sigma_down(scheduler, t_idx)
                #     scaled_grad = scaled_grad * (sigma_up ** 2)
                # # pred_noise = pred_noise + self.gradient_guidance * grad * scale_factor / (dt + 1e-7)
                # pred_noise = pred_noise + self.gradient_guidance * scaled_grad
                # return pred_noise
                pred_noise = pred_noise + self.gradient_guidance * grad
                return pred_noise
        else:
            raise NotImplementedError(f"Proposal distribution {self.proposal_distribution} not implemented in FKD.")
        

# if __name__ == "__main__":

#     # Demonstration of FKD resampling step
#     import matplotlib.pyplot as plt
#     import random

#     # set seed
#     random.seed(0)

#     # 1x1 pixel images
#     num_particles = 8
#     x0s = torch.rand(num_particles, 1, 1)

#     # reward darker images
#     reward_function = lambda x: -0.5 * x.sum(dim=(1, 2))

#     # Define the FKD steering mechanism
#     fkds = FKD(
#         potential_type=PotentialType.DIFF,
#         lmbda=10.0,
#         num_particles=num_particles,
#         adaptive_resampling=False,
#         resample_frequency=1,
#         resampling_t_start=-1,
#         resampling_t_end=100,
#         time_steps=100,
#         reward_fn=lambda x: reward_function(x),
#         device=torch.device('cpu'),
#     )

#     # Define the sampling index
#     sampling_idx = 0

#     # Perform resampling
#     resampled_latents, resampled_images = fkds.resample(
#         sampling_idx=sampling_idx,
#         latents=x0s,
#         x0_preds=x0s,
#     )

#     plt.rc('text', usetex=True)
#     fig, axs = plt.subplots(2, num_particles)

#     axs[0, 0].set_title('Initial')
#     axs[1, 0].set_title('Resampled')

#     for i in range(num_particles):
#         axs[0, i].imshow(x0s[i].detach().numpy(), cmap='gray', vmin=0, vmax=1)
#         axs[1, i].imshow(
#             resampled_images[i].detach().numpy(), cmap='gray', vmin=0, vmax=1
#         )

#         axs[1, i].axis('off')
#         axs[0, i].axis('off')

#     out_path = 'resampled_examples.png'
#     plt.savefig(out_path)
#     print('Saved resampled examples to:', out_path)