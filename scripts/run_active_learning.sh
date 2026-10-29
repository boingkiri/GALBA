#!/bin/bash
# Runs the generative active learning loop for one experiment config.
#
# Each round consists of
#   1. inference_phase.py : generate new labeled samples with the diffusion model,
#                           guided by the current target model (round 0 builds the initial set)
#   2. training_phase.py  : train the target model on all data generated so far
#
# Usage:
#   bash scripts/run_active_learning.sh <config_name> [seed ...]
#
# Examples:
#   bash scripts/run_active_learning.sh imagenette_GALBA.yaml
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_active_learning.sh imagenette_DP.yaml 42 43 44
#
# Environment variables:
#   CUDA_VISIBLE_DEVICES    GPUs to use (default: 0,1,2,3)
#   NUM_ROUNDS              index of the last round; rounds 0..NUM_ROUNDS are run (default: 15)
#   FREEDOM_GUIDANCE_SCALE  gradient guidance scale (default: 0.05)
#   USE_BF16                run generation with bf16 mixed precision (default: true)
#   EXTRA_OVERRIDES         extra Hydra overrides passed to both phases, e.g. "++train.max_steps=100"
#   PJRT_DEVICE=TPU         run on a Cloud TPU VM instead of GPUs

set -e

if [ $# -lt 1 ]; then
  echo "Usage: bash scripts/run_active_learning.sh <config_name> [seed ...]"
  exit 1
fi

config_name=$1
shift
seeds=("$@")
if [ ${#seeds[@]} -eq 0 ]; then
  seeds=(42)
fi

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
num_devices=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
NUM_ROUNDS=${NUM_ROUNDS:-15}
freedom_guidance_scale=${FREEDOM_GUIDANCE_SCALE:-0.05}
use_bf_16=${USE_BF16:-true}

# Run from the repository root regardless of where the script is called from.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
config_path=$(pwd)/configs

for seed in "${seeds[@]}"
do
  default_root_dir=""

  for i in $(seq 0 $NUM_ROUNDS)
  do
    echo "=== [${config_name}] seed ${seed}, round ${i}/${NUM_ROUNDS} ==="

    # --- Inference (data generation) phase ---
    if [[ $PJRT_DEVICE == "TPU" ]]; then
      inference_code="accelerate launch --tpu --main_training_function inference "
    elif [[ $num_devices -gt 1 ]]; then
      inference_code="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES accelerate launch --multi_gpu --num_processes $num_devices "
    else
      inference_code="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES accelerate launch "
    fi

    if [[ $use_bf_16 == true ]]; then
      inference_code+="--mixed_precision bf16 "
    fi

    inference_code+="inference_phase.py "
    inference_code+="--config-path $config_path "
    inference_code+="--config-name $config_name "
    inference_code+="++num_devices=$num_devices "
    inference_code+="++generative_oracle.freedom_guidance_scale=$freedom_guidance_scale "
    inference_code+="++default_root_dir=$default_root_dir "
    inference_code+="++seed=$seed "
    inference_code+="${EXTRA_OVERRIDES} "

    # inference_phase.py prints the run directory as `default_root_dir: <path>`;
    # it is created in the first round and reused afterwards.
    # (captured first so that `set -e` stops the loop if the inference phase fails)
    inference_stdout=$(eval $inference_code)
    echo "$inference_stdout"
    tmp_default_root_dir=$(echo "$inference_stdout" | awk -F': *' '/^default_root_dir:/ {print $2; exit}')
    if [ -z "$default_root_dir" ]; then
      default_root_dir=$tmp_default_root_dir
      echo "Run directory: $default_root_dir"
    fi

    # --- Training phase ---
    training_code="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python training_phase.py "
    training_code+="--config-path $config_path "
    training_code+="--config-name $config_name "
    training_code+="++num_devices=$num_devices "
    training_code+="++default_root_dir=$default_root_dir "
    training_code+="++seed=$seed "
    training_code+="${EXTRA_OVERRIDES} "
    eval $training_code
  done
done
