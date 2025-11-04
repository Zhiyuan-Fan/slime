#!/bin/bash

# 直接从Qwen3预训练模型开始训练 - 跳过SFT阶段
# 支持多轮工具调用的True/False判断任务

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
BASE_DIR="/path/to/your/base/dir"  # 请修改为你的基础目录
MODEL_NAME="qwen3-4b-from-pretrained"

echo "========================================="
echo "🚀 直接从Qwen3预训练模型开始RL训练"
echo "模型: ${MODEL_NAME}"
echo "跳过SFT阶段，直接进行强化学习"
echo "训练数据: 自定义True/False判断数据集"
echo "========================================="

# 数据配置
TRAIN_DATA="./data/custom_dataset/converted_dataset_train.jsonl"
VAL_DATA="./data/custom_dataset/converted_dataset_val.jsonl"

# 检查数据文件
if [ ! -f "$TRAIN_DATA" ]; then
    echo "❌ 训练数据文件不存在: $TRAIN_DATA"
    echo "请先运行数据预处理脚本: python data_preprocessing.py"
    exit 1
fi

# 检查点配置 - 直接从预训练模型开始
CKPT_ARGS=(
    # 🎯 关键配置：直接使用预训练模型
    --hf-checkpoint ${BASE_DIR}/Qwen/Qwen3-4B-Instruct-2507
    --ref-load ${BASE_DIR}/Qwen/Qwen3-4B-Instruct-2507_torch_dist

    # ⚠️ 重要：不设置--load参数，让actor从ref-load初始化
    # --load ${BASE_DIR}/some_sft_checkpoint/  # 注释掉，直接从预训练开始

    # 保存训练后的模型
    --save ${BASE_DIR}/custom_model_from_pretrained/
    --save-interval 20
    --rotary-base 5000000
)

# 模型配置（基于Qwen3-4B）
source "${SCRIPT_DIR}/../../scripts/models/qwen3-4B.sh"

# 性能配置 (4GPU标准)
PERF_ARGS=(
    --tensor-model-parallel-size 2
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1

    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1

    --use-dynamic-batch-size
    --max-tokens-per-gpu 8192
)

# GRPO算法配置 - 从预训练开始的特殊设置
GRPO_ARGS=(
    --advantage-estimator grpo
    --use-kl-loss  # 使用KL散度约束，防止过度偏离预训练模型
    --kl-loss-coef 0.02  # 稍微增加KL系数，保持与预训练模型的相似性
    --kl-loss-type low_var_kl
    --entropy-coef 0.01  # 保持一定的探索性
    --eps-clip 0.2
    --eps-clip-high 0.28
)

# 优化器配置 - 适合从预训练开始的学习率
OPTIMIZER_ARGS=(
    --optimizer adamw
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-5
    --lr 3e-6  # 较小的学习率，避免破坏预训练知识
    --min-lr 1e-7
    --lr-decay-style cosine
    --lr-warmup-iters 100  # 更长的warmup
    --clip-grad 1.0
    --weight-decay 0.1
)

# SGLang推理配置
SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 2
    --sglang-mem-fraction-static 0.7
    --sglang-disable-radix-cache
)

# 训练配置 - 从预训练开始的特殊设置
ROLLOUT_ARGS=(
    --prompt-data ${TRAIN_DATA}
    --input-key prompt
    --label-key label
    --apply-chat-template
    --rollout-shuffle
    --reward-key score
    --num-rollout 2000  # 适中的rollout数量
    --rollout-batch-size 24  # 稍小的批次，更稳定
    --n-samples-per-prompt 6  # 每个提示多采样，增加数据多样性
    --rollout-max-response-len 1024
    --rollout-temperature 0.8  # 适中的温度

    --global-batch-size 144  # 6 * 24 = 144
    --balance-data
)

# 评估配置
EVAL_ARGS=(
    --eval-interval 20
    --eval-prompt-data validation ${VAL_DATA}
    --n-samples-per-eval-prompt 8
    --eval-max-response-len 1024
    --eval-top-p 0.7
)

# Wandb配置
WANDB_ARGS=(
    --use-wandb
    --wandb-project "qwen3-from-pretrained-tools"
    --wandb-group "direct-rl-training"
    --wandb-name "${MODEL_NAME}-$(date +%Y%m%d-%H%M%S)"
)

# 其他配置
MISC_ARGS=(
    --seed 42
    --attention-dropout 0.0
    --hidden-dropout 0.0

    # 从预训练开始的稳定性设置
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash

    --use-rotary-position-embeddings
    --normalization RMSNorm
    --swiglu
    --untie-embeddings-and-output-weights
    --disable-bias-linear
    --rotary-base 5000000

    --fp16
    --apply-layernorm-1p
)

# 自定义函数配置 - 支持工具调用
CUSTOM_ARGS=(
    --custom-generate-function-path generate_with_tools.generate
    --custom-rm-path generate_with_tools.reward_func
)

echo "📋 配置检查："
echo "  预训练模型: ${BASE_DIR}/Qwen/Qwen3-4B-Instruct-2507"
echo "  训练数据: ${TRAIN_DATA}"
echo "  保存路径: ${BASE_DIR}/custom_model_from_pretrained/"
echo "  学习率: 3e-6 (适合预训练起点)"
echo "  KL约束: 0.02 (防止过度偏离)"
echo ""

# 启动Ray集群
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 4 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# 环境配置
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}:$(dirname ${SCRIPT_DIR}):$(dirname $(dirname ${SCRIPT_DIR}))\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
  }
}"

echo "🚀 开始直接从预训练模型进行RL训练..."

# 执行训练
ray job submit --address="http://127.0.0.1:8265" \\
   --runtime-env-json="${RUNTIME_ENV_JSON}" \\
   -- python3 train.py \\
   --actor-num-nodes 1 \\
   --actor-num-gpus-per-node 4 \\
   --colocate \\
   ${MODEL_ARGS[@]} \\
   ${CKPT_ARGS[@]} \\
   ${ROLLOUT_ARGS[@]} \\
   ${OPTIMIZER_ARGS[@]} \\
   ${GRPO_ARGS[@]} \\
   ${WANDB_ARGS[@]} \\
   ${PERF_ARGS[@]} \\
   ${EVAL_ARGS[@]} \\
   ${SGLANG_ARGS[@]} \\
   ${MISC_ARGS[@]} \\
   ${CUSTOM_ARGS[@]}

echo "========================================="
echo "✅ 从预训练模型的RL训练完成！"
echo "📁 模型保存在: ${BASE_DIR}/custom_model_from_pretrained/"
echo "📊 Wandb项目: qwen3-from-pretrained-tools"
echo "🎯 模型现在具备工具调用推理能力！"
echo "========================================="