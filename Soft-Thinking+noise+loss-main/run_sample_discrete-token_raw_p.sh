#!/usr/bin/env bash

set -e

model_name="/workspace/yiqiuguo/SofT-GRPO-master/saved_weight/Qwen3-1.7B-Base-SoftGRPO-157"

datasets=(
  # aime2024
  # aime2025
  math500
  amc23
)
CUDA_VISIBLE_DEVICES=0,1,2,3
for i in "${!datasets[@]}"; do
  # CUDA_VISIBLE_DEVICES=$i \
  python run_sglang_softthinking.py \
    --dataset "${datasets[$i]}" \
    --model_name "$model_name" \
    --max_topk 30 \
    --max_generated_tokens 30000 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 30 \
    --min_p 0.0 \
    --after_thinking_temperature 0.6 \
    --after_thinking_top_p 0.95 \
    --after_thinking_top_k 30 \
    --after_thinking_min_p 0.0 \
    --mem_fraction_static 0.9 \
    --start_idx 0 \
    --end_idx 1000 \
    --num_gpus 4 \
    --num_samples 32 \
    > "log_${datasets[$i]}.out" 2>&1
done

    # --enable_soft_thinking \
    # --add_noise_gumbel_softmax \
    # --gumbel_softmax_temperature 0.5 \
    # --noise_factor 1 \
echo "✅ All jobs submitted."