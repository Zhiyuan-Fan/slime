#!/bin/bash

# 不使用工具调用的True/False判断训练脚本
# 纯语言推理版本，用于对比工具调用的效果

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
BASE_DIR="/path/to/your/base/dir"  # 请修改为你的基础目录
MODEL_NAME="qwen3-4b-no-tools"

echo "========================================="
echo "🧠 启动纯语言推理训练 (无工具调用)"
echo "模型: ${MODEL_NAME}"
echo "训练方式: 纯语言推理，不使用代码执行工具"
echo "对比目标: 验证工具调用的效果提升"
echo "========================================="

# 数据配置 - 使用相同的数据集
TRAIN_DATA="./data/custom_dataset/converted_dataset_train.jsonl"
VAL_DATA="./data/custom_dataset/converted_dataset_val.jsonl"

# 检查数据文件
if [ ! -f "$TRAIN_DATA" ]; then
    echo "❌ 训练数据文件不存在: $TRAIN_DATA"
    echo "请先运行数据预处理脚本: python data_preprocessing.py"
    exit 1
fi

# 检查点配置
CKPT_ARGS=(
    --hf-checkpoint ${BASE_DIR}/Qwen/Qwen3-4B-Instruct-2507
    --ref-load ${BASE_DIR}/Qwen/Qwen3-4B-Instruct-2507_torch_dist
    # 不设置--load，直接从预训练模型开始
    --save ${BASE_DIR}/custom_model_no_tools/
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

# GRPO算法配置
GRPO_ARGS=(
    --advantage-estimator grpo
    --use-kl-loss
    --kl-loss-coef 0.02  # 保持与预训练模型的一致性
    --kl-loss-type low_var_kl
    --entropy-coef 0.01
    --eps-clip 0.2
    --eps-clip-high 0.28
)

# 优化器配置 - 纯推理可能需要更多训练
OPTIMIZER_ARGS=(
    --optimizer adamw
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-5
    --lr 4e-6  # 稍微提高学习率，因为需要学习更复杂的推理
    --min-lr 1e-7
    --lr-decay-style cosine
    --lr-warmup-iters 150  # 更长的warmup
    --clip-grad 1.0
    --weight-decay 0.1
)

# SGLang推理配置
SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 2
    --sglang-mem-fraction-static 0.7
    --sglang-disable-radix-cache
)

# 训练配置 - 纯推理的特殊设置
ROLLOUT_ARGS=(
    --prompt-data ${TRAIN_DATA}
    --input-key prompt
    --label-key label
    --apply-chat-template
    --rollout-shuffle
    --reward-key score
    --num-rollout 2500  # 更多rollout，因为纯推理需要更多样本
    --rollout-batch-size 20
    --n-samples-per-prompt 8  # 更多采样，增加推理多样性
    --rollout-max-response-len 1536  # 增加长度，允许更详细的推理
    --rollout-temperature 0.85  # 稍高温度，鼓励多样化推理

    --global-batch-size 160  # 8 * 20 = 160
    --balance-data
)

# 评估配置
EVAL_ARGS=(
    --eval-interval 25  # 更频繁评估，关注纯推理的学习曲线
    --eval-prompt-data validation ${VAL_DATA}
    --n-samples-per-eval-prompt 10
    --eval-max-response-len 1536
    --eval-top-p 0.8  # 稍高，鼓励更多样的推理路径
)

# Wandb配置
WANDB_ARGS=(
    --use-wandb
    --wandb-project "qwen3-reasoning-comparison"
    --wandb-group "no-tools-baseline"
    --wandb-name "${MODEL_NAME}-$(date +%Y%m%d-%H%M%S)"
    # 添加tags标识这是无工具版本
    --wandb-tags "no_tools,pure_reasoning,baseline"
)

# 其他配置
MISC_ARGS=(
    --seed 42
    --attention-dropout 0.0
    --hidden-dropout 0.0
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

# 自定义函数配置 - 使用无工具版本
CUSTOM_ARGS=(
    --custom-generate-function-path generate_without_tools.generate
    --custom-rm-path generate_without_tools.reward_func
)

echo "📋 纯推理训练配置："
echo "  🧠 推理模式: 纯语言推理，无工具辅助"
echo "  📚 数据集: ${TRAIN_DATA}"
echo "  🎯 目标: 通过语言推理解决True/False判断"
echo "  📊 对比: 将与工具调用版本进行效果对比"
echo "  💡 奖励: 重点奖励推理过程和逻辑结构"
echo "  🔍 评估: 关注推理质量、知识运用等指标"
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

echo "🧠 开始纯语言推理训练..."
echo "⏱️  预计训练时间: 比工具版本更长（需要更多样本学习复杂推理）"

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
echo "✅ 纯推理训练完成！"
echo "📁 模型保存: ${BASE_DIR}/custom_model_no_tools/"
echo "📊 Wandb项目: qwen3-reasoning-comparison"
echo "🧠 模型现在具备纯语言推理能力"
echo "🔍 接下来可以与工具调用版本进行对比分析"
echo "========================================="