# 自定义True/False判断任务

这个示例展示如何在slime框架中添加自定义的True/False判断数据集，并进行强化学习训练。

## 数据集格式

输入数据格式为JSONL，每行包含：
```json
{
  "prompt": "请判断以下命题是否正确：所有的质数都是奇数。请在\\boxed{}中给出你的答案（true或false）。",
  "label": "false"
}
```

## 文件说明

### 1. `data_preprocessing.py`
数据预处理脚本，功能包括：
- 将原始数据集转换为slime标准格式
- 数据验证和清洗
- 数据集分割（训练/验证/测试）

### 2. `generate_with_custom_reward.py`
自定义生成和奖励函数：
- `extract_boxed_answer()`: 提取`\\boxed{}`格式的答案
- `normalize_answer()`: 标准化答案格式
- `generate()`: 自定义生成函数
- `reward_func()`: 自定义奖励函数，支持详细的评估指标

### 3. `train_custom_dataset.sh`
完整的训练脚本配置，包含所有必要参数。

## 使用步骤

### 步骤1: 准备数据
```bash
# 1. 将你的数据集放在合适位置，例如：
# ./data/raw/your_dataset.jsonl

# 2. 运行数据预处理
cd examples/custom_dataset/
python data_preprocessing.py
```

请修改 `data_preprocessing.py` 中的路径：
```python
input_file = "path/to/your/input_dataset.jsonl"  # 修改为你的数据集路径
```

### 步骤2: 配置模型和路径
修改 `train_custom_dataset.sh` 中的配置：
```bash
BASE_DIR="/path/to/your/base/dir"  # 修改为你的基础目录
```

确保以下文件存在：
- 预训练模型: `${BASE_DIR}/Qwen/Qwen3-4B-Instruct-2507`
- Torch格式模型: `${BASE_DIR}/Qwen/Qwen3-4B-Instruct-2507_torch_dist`

### 步骤3: 运行训练
```bash
# 给脚本执行权限
chmod +x train_custom_dataset.sh

# 启动训练
./train_custom_dataset.sh
```

## 奖励函数设计

奖励函数考虑以下因素：

1. **基础准确性** (+1.0 正确, -0.5 错误, -1.0 无答案)
2. **格式遵循** (+0.1 正确使用`\\boxed{}`, -0.2 未使用)
3. **长度惩罚** (-0.1 过短或过长的回答)

最终奖励范围: [-2.0, 2.0]

## 评估指标

系统会自动计算以下指标：
- `accuracy`: 答案准确率
- `mean_reward`: 平均奖励值
- `format_compliance`: 格式遵循率
- `response_length`: 平均回答长度

## 自定义配置

### 修改奖励函数
在 `generate_with_custom_reward.py` 的 `reward_func()` 中修改奖励逻辑：
```python
# 调整奖励值
base_reward = 2.0 if normalized_prediction == ground_truth else -1.0

# 添加新的奖励因子
reasoning_bonus = 0.2 if "因为" in full_response else 0.0
```

### 修改生成提示
在 `generate()` 函数中调整系统提示：
```python
system_prompt = (
    "你的自定义系统提示..."
)
```

### 调整训练参数
在 `train_custom_dataset.sh` 中修改：
```bash
# 学习率
--lr 1e-5

# 温度参数
--temperature 0.8

# 最大生成长度
--max-new-tokens 256
```

## 故障排除

### 常见问题

1. **数据格式错误**
   - 检查JSONL格式是否正确
   - 确保每行都有`prompt`和`label`字段

2. **模型加载失败**
   - 验证模型路径是否正确
   - 检查torch_dist格式模型是否存在

3. **内存不足**
   - 减少`--max-tokens-per-gpu`
   - 调整并行配置参数

4. **奖励函数错误**
   - 检查导入路径是否正确
   - 确保函数签名与slime要求一致

### 调试技巧

1. **测试奖励函数**
```bash
cd examples/custom_dataset/
python generate_with_custom_reward.py
```

2. **查看生成样本**
在训练过程中检查wandb日志中的样本输出。

3. **验证数据预处理**
检查转换后的数据格式：
```bash
head -5 ./data/custom_dataset/converted_dataset.jsonl
```

## 扩展功能

### 支持多选题
修改 `normalize_answer()` 函数以支持A/B/C/D选项：
```python
def normalize_answer(answer: str) -> str:
    answer = answer.upper().strip()
    if answer in ["A", "B", "C", "D"]:
        return answer
    return ""
```

### 添加置信度评估
在奖励函数中考虑模型的置信度：
```python
confidence_patterns = ["确定", "肯定", "可能", "不确定"]
confidence_bonus = 0.1 if any(p in full_response for p in confidence_patterns) else 0.0
```

## 性能优化

1. **数据缓存**: 预处理大型数据集时使用缓存
2. **并行训练**: 根据GPU数量调整并行参数
3. **内存优化**: 使用梯度检查点减少内存使用

## 参考文档

- [slime官方文档](https://thudm.github.io/slime/)
- [Quick Start指南](../../docs/en/get_started/quick_start.md)
- [参数说明](../../docs/en/get_started/usage.md)