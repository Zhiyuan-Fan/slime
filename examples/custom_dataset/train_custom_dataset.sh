#!/bin/bash

# 自定义True/False判断任务的训练脚本（支持多轮工具调用）
# 使用slime框架进行强化学习训练

set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# 基础配置
BASE_DIR="/path/to/your/base/dir"  # 请修改为你的基础目录
MODEL_NAME="qwen3-4b"  # 可根据需要修改

# 数据配置
TRAIN_DATA="./data/custom_dataset/converted_dataset_train.jsonl"
VAL_DATA="./data/custom_dataset/converted_dataset_val.jsonl"

echo "========================================="
echo "启动支持工具调用的True/False推理训练"
echo "模型: ${MODEL_NAME}"
echo "训练数据: ${TRAIN_DATA}"
echo "基础目录: ${BASE_DIR}"
echo "========================================="

# 检查点配置
CKPT_ARGS=(
    # HF checkpoint，SGLang需要，也从这里读取tokenizer
    --hf-checkpoint ${BASE_DIR}/Qwen/Qwen3-4B-Instruct-2507
    # 参考模型加载目录
    --ref-load ${BASE_DIR}/Qwen/Qwen3-4B-Instruct-2507_torch_dist
    # Actor模型加载目录，如果为空会从ref_load读取
    --load ${BASE_DIR}/custom_model_checkpoints/
    # 保存目录
    --save ${BASE_DIR}/custom_model_checkpoints/
    --save-interval 50
)

# 模型配置（基于Qwen3-4B）
source "${SCRIPT_DIR}/../../scripts/models/qwen3-4B.sh"

# 性能配置 (4张卡配置)
PERF_ARGS=(
    --tensor-model-parallel-size 2
    --sequence-parallel
    --pipeline-model-parallel-size 1  # 修正: 使用PP=1，总共2张卡用于训练

    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1

    # 动态批次大小
    --use-dynamic-batch-size
    --max-tokens-per-gpu 8192
)

# GRPO算法配置
GRPO_ARGS=(
    --advantage-estimator grpo
    --use-kl-loss
    --kl-loss-coef 0.01
    --kl-loss-type low_var_kl
    --entropy-coef 0.01
    --eps-clip 0.2
    --eps-clip-high 0.28
)

# 优化器配置
OPTIMIZER_ARGS=(
    --optimizer adamw
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-5
    --lr 5e-6
    --min-lr 1e-6
    --lr-decay-style cosine
    --lr-warmup-iters 100
    --clip-grad 1.0
    --weight-decay 0.1
)

# SGLang推理配置
SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 2
    --sglang-mem-fraction-static 0.8
    --sglang-disable-radix-cache
)

# 训练配置
TRAIN_ARGS=(
    # 基础训练参数
    --train-iters 1000
    --eval-interval 100
    --eval-iters 10
    --log-interval 10

    # 批次配置
    --micro-batch-size 4
    --global-batch-size 32

    # 数据配置
    --data-path ${TRAIN_DATA}
    --prompt-key prompt
    --label-key label

    # 推理配置
    --temperature 0.7
    --top-p 0.9
    --max-new-tokens 512

    # 自定义函数配置（支持工具调用）
    --generate-function examples.custom_dataset.generate_with_tools.generate
    --reward-function examples.custom_dataset.generate_with_tools.reward_func
)

# 其他参数
MISC_ARGS=(
    --seed 42
    --dataloader-type external
    --no-query-key-layer-scaling
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --use-rotary-position-embeddings
    --normalization RMSNorm
    --swiglu
    --untie-embeddings-and-output-weights
    --disable-bias-linear
    --rotary-base 5000000

    # 混合精度
    --fp16
    --apply-layernorm-1p

    # 日志和监控
    --wandb-project "custom-true-false-reasoning-with-tools"
    --wandb-name "${MODEL_NAME}-tool-enabled-$(date +%Y%m%d-%H%M%S)"
)

# 启动Ray集群（如果需要）
ray start --head --port=6379 --num-gpus=$(nvidia-smi -L | wc -l) --disable-usage-stats || echo "Ray已经在运行"

# 执行训练
cd ${SCRIPT_DIR}/../..

python -u train.py \\
    ${CKPT_ARGS[@]} \\
    ${MODEL_ARGS[@]} \\
    ${PERF_ARGS[@]} \\
    ${GRPO_ARGS[@]} \\
    ${OPTIMIZER_ARGS[@]} \\
    ${SGLANG_ARGS[@]} \\
    ${TRAIN_ARGS[@]} \\
    ${MISC_ARGS[@]}

# 等待训练完成并显示结果
wait

echo "========================================="
echo "工具集成推理训练完成！"
echo "检查点保存在: ${BASE_DIR}/custom_model_checkpoints/"
echo "Wandb项目: custom-true-false-reasoning-with-tools"
echo "========================================="