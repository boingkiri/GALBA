# GALBA: Generative Active Learning via Bayesian Acquisition

Official implementation of
**Generative Active Learning via Bayesian Acquisition for Improving the Efficiency of Synthetic Data** (NeurIPS 2026).

**Jeongjun Lee, Gyeonghoon Ko, Juho Lee**

<!-- TODO: paper / arXiv link -->

A target model is trained only on images synthesized by a text-to-image diffusion model.
Training proceeds in active learning rounds: in each round the diffusion model generates a new batch of
labeled samples, steered by the current target model towards samples that are informative for it, and
the target model is retrained on everything generated so far.
**GALBA** steers generation with a Bayesian acquisition function (BALD) using
Feynman–Kac particle resampling during diffusion sampling.

---

## Contents

- [Methods](#methods)
- [Installation](#installation)
- [Data and pretrained models](#data-and-pretrained-models)
- [Running experiments](#running-experiments)
- [Outputs and logging](#outputs-and-logging)
- [Configuration](#configuration)
- [Repository structure](#repository-structure)
- [Citation](#citation)
- [License](#license)

## Methods

Every dataset has three experiment configs that are identical except for how generation is guided.

| Config | Guidance during generation |
|---|---|
| `*_no_guidance` | Plain class-conditioned generation, no target-model guidance |
| `*_DP` | Gradient guidance on the denoising trajectory that increases the predictive entropy of the target model (scale 0.05) |
| `*_GALBA` | **Ours.** Feynman–Kac particle resampling (4 particles, `max` potential) with a BALD reward computed in closed form from a class-wise last-layer Laplace approximation of the target model |

The generator is Stable Diffusion v1.5 prompted with the class names, and the target model is a ViT-B trained from scratch.

The first round of every run generates the initial labeled set without guidance, so all three methods
start from the same kind of data.

## Installation

Tested on Ubuntu with Python 3.10, CUDA 12.8 and NVIDIA RTX 3090 (24 GB) GPUs.

```bash
git clone <this repository>
cd <repository>

conda create -n galba python=3.10 -y
conda activate galba

# Install PyTorch for your CUDA version first (https://pytorch.org/get-started/locally/), e.g.
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

Runs are logged to [Weights & Biases](https://wandb.ai). Log in once with `wandb login`,
or run without an account with `export WANDB_MODE=offline`.

## Data and pretrained models

Everything is downloaded automatically on first use. All relative paths in the configs
are resolved against the repository root.

| Dataset | Source | Location |
|---|---|---|
| ImageNette | `torchvision.datasets.Imagenette` (full size) | `data/imagenette/` |
| ImageNet-100 | HuggingFace Hub [`clane9/imagenet-100`](https://huggingface.co/datasets/clane9/imagenet-100) | HuggingFace cache (`HF_HOME`) |

If you already have ImageNette, link it instead of downloading again:

```bash
ln -s /path/to/imagenette data/imagenette   # contains imagenette2/
```

The target model is trained only on generated images (`num_initial_labeled_real_samples: 0`);
model selection and evaluation use the real validation sets.

Pretrained models (downloaded from the HuggingFace Hub):

| Model | Used for |
|---|---|
| [`stable-diffusion-v1-5/stable-diffusion-v1-5`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) | generator |
| [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32) | selecting the generated holdout set used by the acquisition functions |

## Running experiments

Launch one experiment with

```bash
bash scripts/run_active_learning.sh <config> [seed ...]
```

Examples:

```bash
# GALBA on ImageNette, seed 42, GPUs 0-3
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_active_learning.sh imagenette_GALBA.yaml 42

# The two baselines with three seeds each
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_active_learning.sh imagenette_no_guidance.yaml 42 43 44
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_active_learning.sh imagenette_DP.yaml 42 43 44
```

| Dataset | Configs |
|---|---|
| ImageNette | `imagenette_no_guidance.yaml`, `imagenette_DP.yaml`, `imagenette_GALBA.yaml` |
| ImageNet-100 | `imagenet_100_no_guidance.yaml`, `imagenet_100_DP.yaml`, `imagenet_100_GALBA.yaml` |

The script runs rounds `0 .. NUM_ROUNDS` (16 rounds by default). Each round has two phases, each a separate process:

1. **Generation** (`inference_phase.py`, launched with `accelerate` on all GPUs):
   loads the latest target model checkpoint and generates
   `generative_oracle.num_generated_sample_per_cycle` new labeled images with the chosen guidance.
   Round 0 generates the initial set of `num_initial_labeled_generated_samples` images without guidance.
2. **Training** (`training_phase.py`, PyTorch Lightning DDP):
   trains the target model on all images generated so far, validating on the real validation set
   with early stopping, and saves the best checkpoint.

Environment variables of `scripts/run_active_learning.sh`:

| Variable | Default | Description |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3` | GPUs to use; the number of processes equals the number of GPUs |
| `NUM_ROUNDS` | `15` | index of the last round |
| `FREEDOM_GUIDANCE_SCALE` | `0.05` | scale of the gradient guidance (`DP`) |
| `USE_BF16` | `true` | bf16 autocast in the generation phase |
| `EXTRA_OVERRIDES` | | extra [Hydra overrides](https://hydra.cc/docs/advanced/override_grammar/basic/) passed to both phases |
| `PJRT_DEVICE` | | set to `TPU` to run on a Cloud TPU VM |

### Quick test

To check the whole pipeline in a few minutes, run two rounds with tiny generation and training budgets:

```bash
export NUM_ROUNDS=1 WANDB_MODE=offline
export EXTRA_OVERRIDES="++train.max_steps=2 ++train.batch_size=4 ++generative_oracle.batch_size=1 \
  ++num_initial_labeled_generated_samples=40 ++generative_oracle.num_generated_sample_per_cycle=40 \
  ++dataset.num_holdout_samples=20 ++exp_name=quick_test"
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_active_learning.sh imagenette_GALBA.yaml 42
```

The number of generated samples must be divisible by the number of classes and the number of GPUs.

### Hardware

The ImageNette experiments were run on 4 × RTX 3090 (24 GB).
The ImageNet-100 experiments were run on Cloud TPU VMs (`PJRT_DEVICE=TPU`, with `torch_xla`);
on TPU, generated data and checkpoints are stored in a Google Cloud Storage bucket
given by `GCS_BUCKET_NAME`. The same configs also run on GPUs.

## Outputs and logging

Each run writes to `generative_active_learning_<dataset>/<YYMMDD_HHMMSS>_<exp_name>/seed_<seed>/`:

```
seed_42/
├── initial_generated_data/      # round 0: initial labeled set (images + labels)
├── cycle_0/
│   ├── epoch=..-val_accuracy=..ckpt   # best target model of cycle 0
│   ├── generated_dataset/       # images generated with the cycle-0 model (used from cycle 1 on)
│   └── cycle_status.yaml        # which phases of this cycle are finished
├── cycle_1/
└── ...
```

The validation accuracy of each cycle is logged to W&B as `Cycle_<k>/val_accuracy`, grouped by `exp_name`.
Hydra logs go to `outputs/`.

## Configuration

Configs are managed with [Hydra](https://hydra.cc). An experiment config in `configs/` composes the
component configs below and overrides some of their values.

```
configs/
├── <dataset>_<method>.yaml   # experiment configs (6)
├── dataset/                  # data paths and transforms
├── target_model/             # vit_b
├── train/                    # optimizer, LR schedule, number of steps
├── callbacks/                # checkpointing / early stopping metric
├── generative_oracle/        # diffusion model, sampler, FK steering parameters
└── guidance_model/           # acquisition function: no_guidance, entropy, bald
```

Main settings:

| | ImageNette | ImageNet-100 |
|---|---|---|
| Initial generated set | 3,200 | 32,000 |
| Generated per round | 1,600 | 16,000 |
| Sampler | Euler ancestral, 50 steps, CFG 3 | Euler ancestral, 50 steps, CFG 3 |
| Optimizer | AdamW, lr 1e-4, batch 128 | AdamW, lr 1e-3, batch 1024 |
| Max steps per round | 25k | 50k |
| GALBA: λ / particles | 16 / 4 | 24 / 4 |

Any value can be changed from the command line through `EXTRA_OVERRIDES`, e.g.
`EXTRA_OVERRIDES="++generative_oracle.fkd_args.lmbda=8.0"`.

## Repository structure

```
inference_phase.py                     # generation phase of one round
training_phase.py                      # training phase of one round
scripts/run_active_learning.sh         # runs all rounds of an experiment
generative_oracle/
├── generative_oracle_container.py     # generation loop and saving of generated data
├── classification_generative_oracle.py  # Stable Diffusion oracle
├── guidance_model_container.py        # acquisition functions: entropy, BALD
├── FKD/                               # Feynman-Kac particle resampling
└── stable_diffusion/                  # Stable Diffusion pipeline with gradient guidance
models/
├── target_model_framework.py          # LightningModule (EMA, optimizers, validation metrics)
└── classification/                    # ViT-B
utils/                                 # data modules, trainer / callbacks, cycle bookkeeping
configs/                               # Hydra configs
data/imagenet_100/imagenet100_label.yaml  # ImageNet-100 class names used for the prompts
```

## Citation

```bibtex
@inproceedings{lee2026generative,
  title     = {Generative Active Learning via Bayesian Acquisition for Improving the Efficiency of Synthetic Data},
  author    = {Lee, Jeongjun and Ko, Gyeonghoon and Lee, Juho},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## License

This code is released under the [Apache License 2.0](LICENSE).

## Acknowledgements

This code builds on [diffusers](https://github.com/huggingface/diffusers),
[Feynman-Kac steering](https://github.com/zacharyhorvitz/Fk-Diffusion-Steering) and
[Feedback-guided data synthesis](https://github.com/facebookresearch/feedback-guided-data-synthesis).
Files adapted from diffusers keep their original Apache-2.0 license headers.
