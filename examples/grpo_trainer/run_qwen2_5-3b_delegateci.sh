#!/usr/bin/env bash
# GRPO training entrypoint for the CI-guided query rewriter.
#   - 4 x A100 (80G), Qwen2.5-3B-Instruct, GRPO
#   - dataset: DelegateCI-Bench (train split), see README for the parquet format
#
# Configurable environment variables (all have sensible defaults):
#   MODEL_PATH    : path or HuggingFace id of the base model
#   DATA_DIR      : directory containing {train,test}.parquet
#   PROJECT_ROOT  : root of this repository (auto-detected by default)
#   RAY_TMPDIR    : optional override for Ray temp dir
#
# Example:
#   MODEL_PATH=Qwen/Qwen2.5-3B-Instruct \
#   DATA_DIR=./data/delegateci \
#   bash examples/grpo_trainer/run_qwen2_5-3b_delegateci.sh

set -x

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

# Resolve repository root (the directory two levels above this script).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}
DATA_DIR=${DATA_DIR:-${PROJECT_ROOT}/data/delegateci}
REWARD_FN=${REWARD_FN:-${PROJECT_ROOT}/verl/utils/reward_score/priv_batch.py}

mkdir -p logs trajectory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/qwen2.5_3b_grpo_priv_${TIMESTAMP}.log"
TRAJECTORY_FILE="trajectory/qwen2.5_3b_grpo_priv_${TIMESTAMP}.jsonl"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/test.parquet \
    data.train_batch_size=24 \
    data.max_prompt_length=1024 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=True \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=24 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=6 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.005 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=24 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=24 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=batch \
    custom_reward_function.path=${REWARD_FN} \
    custom_reward_function.name=compute_score \
    +custom_reward_function.reward_kwargs.bert_batch_size=128 \
    +custom_reward_function.reward_kwargs.log_jsonl_path=$TRAJECTORY_FILE \
    trainer.critic_warmup=0 \
    trainer.logger=['swanlab'] \
    trainer.project_name='need2know' \
    trainer.experiment_name='qwen2.5_3b_grpo' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=80 \
    trainer.log_val_generations=10 \
    trainer.total_epochs=10 $@ 2>&1 | tee "$LOG_FILE"