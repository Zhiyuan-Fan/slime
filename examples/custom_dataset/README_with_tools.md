# 支持工具调用的True/False判断任务

这个示例展示如何在slime框架中实现支持**多轮工具调用**的True/False判断任务。模型可以在推理过程中生成并执行Python代码来辅助判断。

## 🚀 核心特性

### 1. 多轮工具调用
- **代码生成**: 模型可以生成Python代码来验证命题
- **安全执行**: 在沙箱环境中安全执行代码
- **多轮交互**: 支持最多8轮工具调用
- **智能推理**: 基于执行结果进行进一步推理

### 2. 智能奖励机制
- **基础准确性**: +1.0 (正确) / -0.5 (错误) / -1.0 (无答案)
- **工具使用奖励**: 合理使用工具获得额外奖励
- **推理质量**: 包含推理过程的回答获得奖励
- **格式合规**: 正确使用`\\boxed{}`格式

### 3. 安全代码执行
- **沙箱环境**: 隔离执行环境，防止恶意代码
- **资源限制**: 内存、时间和CPU限制
- **模块白名单**: 仅允许安全的数学和科学计算库

## 📁 文件结构

```
examples/custom_dataset/
├── tool_sandbox.py              # 安全代码执行沙箱
├── generate_with_tools.py       # 支持工具调用的生成和奖励函数
├── data_preprocessing.py        # 数据预处理脚本
├── train_custom_dataset.sh      # 训练脚本
├── README_with_tools.md         # 本文档
└── data/custom_dataset/         # 数据目录
```

## 🔧 数据集格式

### 输入格式 (JSONL)
```json
{
  "prompt": "请判断：如果一个数是质数，那么它一定是奇数。请分析并在\\boxed{}中给出答案（true或false）。",
  "label": "false"
}
```

### 模型行为示例
```
用户: 请判断：如果一个数是质数，那么它一定是奇数。

模型: 我需要分析这个命题。让我用代码来检查质数的性质。

<tool_call>
{"name": "code_interpreter", "arguments": {"code": "# 检查前10个质数\nprimes = []\nnum = 2\nwhile len(primes) < 10:\n    is_prime = True\n    for i in range(2, int(num**0.5) + 1):\n        if num % i == 0:\n            is_prime = False\n            break\n    if is_prime:\n        primes.append(num)\n    num += 1\n\nprint('前10个质数:', primes)\nprint('奇偶性:', [('奇数' if p % 2 == 1 else '偶数') for p in primes])"}}
</tool_call>

<interpreter>
Output:
前10个质数: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
奇偶性: ['偶数', '奇数', '奇数', '奇数', '奇数', '奇数', '奇数', '奇数', '奇数', '奇数']
</interpreter>

通过代码验证可以看出，质数中包含偶数2。因此"所有质数都是奇数"这个命题是错误的。

\\boxed{false}
```

## 🛠 使用步骤

### 步骤1: 环境准备
```bash
cd slime/examples/custom_dataset/

# 安装依赖
pip install jinja2 psutil

# 确保有足够的系统资源（推荐8GB+ RAM）
```

### 步骤2: 数据预处理
```bash
# 修改 data_preprocessing.py 中的路径
python data_preprocessing.py

# 检查生成的数据
head -5 ./data/custom_dataset/converted_dataset_train.jsonl
```

### 步骤3: 配置训练
修改 `train_custom_dataset.sh` 中的配置：
```bash
BASE_DIR="/path/to/your/models"  # 模型基础目录
TRAIN_DATA="./data/custom_dataset/converted_dataset_train.jsonl"
```

### 步骤4: 启动训练
```bash
chmod +x train_custom_dataset.sh
./train_custom_dataset.sh
```

## ⚙️ 工具调用配置

### 工具沙箱配置
```python
TOOL_CONFIGS = {
    "max_turns": 12,           # 最大对话轮数
    "max_tool_calls": 8,       # 最大工具调用次数
    "tool_concurrency": 16,    # 并发工具执行数
    "python_timeout": 60,      # Python执行超时(秒)
    "python_memory_limit": "2GB"  # 内存限制
}
```

### 允许的Python库
- **数学计算**: `math`, `statistics`, `decimal`, `fractions`
- **科学计算**: `numpy`, `sympy`, `pandas`
- **基础功能**: `random`, `datetime`, `collections`, `itertools`

### 安全限制
- ❌ 文件系统访问 (`open`, `file`)
- ❌ 系统调用 (`os`, `sys`, `subprocess`)
- ❌ 网络访问
- ❌ 危险函数 (`eval`, `exec`, `import`)

## 📊 评估指标

训练过程中会自动计算以下指标：

### 核心指标
- **accuracy**: 答案准确率
- **mean_reward**: 平均奖励值
- **format_compliance**: `\\boxed{}`格式遵循率

### 工具使用指标
- **tool_usage_rate**: 工具使用率
- **avg_tool_calls**: 平均工具调用次数
- **reasoning_rate**: 包含推理过程的比例

## 🎯 奖励机制详解

### 奖励组成
```python
final_reward = base_reward + tool_reward + format_bonus + reasoning_bonus + length_penalty
```

### 各项奖励说明

1. **基础奖励** (`base_reward`)
   - 正确答案: +1.0
   - 错误答案: -0.5
   - 无有效答案: -1.0

2. **工具使用奖励** (`tool_reward`)
   - 正确答案 + 使用工具: +0.1 到 +0.3
   - 错误答案 + 使用工具: +0.05 到 +0.1 (鼓励尝试)
   - 错误答案 + 未使用工具: -0.1

3. **格式奖励** (`format_bonus`)
   - 使用 `\\boxed{}`: +0.1
   - 未使用格式: -0.2

4. **推理奖励** (`reasoning_bonus`)
   - 包含推理过程: +0.1

5. **长度惩罚** (`length_penalty`)
   - 过短回答 (<30字符): -0.1
   - 过长回答 (>2000字符): -0.1

## 🔧 自定义配置

### 修改工具配置
在 `tool_sandbox.py` 中调整：
```python
# 增加允许的库
self.allowed_modules.add("matplotlib")  # 添加绘图库

# 调整资源限制
TOOL_CONFIGS["python_timeout"] = 120   # 增加超时时间
```

### 调整奖励策略
在 `generate_with_tools.py` 的 `reward_func()` 中：
```python
# 增加工具使用奖励
if tool_call_count > 0 and accuracy == 1.0:
    tool_reward = min(0.5, tool_call_count * 0.15)  # 更高奖励

# 添加创新奖励
creativity_bonus = 0.1 if "创新" in full_response else 0.0
```

### 修改提示模板
在 `format_conversation_with_tools()` 中自定义系统提示：
```python
system_content = (
    "你是一个专业的逻辑分析师。"
    "对于复杂问题，请使用代码工具进行验证。"
    "提供详细的分析过程，最后用\\boxed{true/false}给出结论。"
)
```

## 📈 训练监控

### Wandb集成
训练过程会自动记录到Wandb:
- 项目名: `custom-true-false-reasoning-with-tools`
- 实时指标监控
- 样本生成质量追踪

### 关键监控指标
- `debug/tool_call_count`: 工具调用统计
- `debug/tools_used`: 工具使用次数
- `accuracy`: 准确率趋势
- `tool_usage_rate`: 工具使用率变化

## 🐛 故障排除

### 常见问题

1. **工具执行超时**
   ```bash
   # 解决方案：增加超时时间
   TOOL_CONFIGS["python_timeout"] = 120
   ```

2. **内存不足**
   ```bash
   # 解决方案：减少并发数
   TOOL_CONFIGS["tool_concurrency"] = 8
   ```

3. **代码安全检查失败**
   ```python
   # 解决方案：检查允许的模块列表
   self.allowed_modules.add("your_module")
   ```

4. **生成函数导入失败**
   ```bash
   # 解决方案：确保正确的Python路径
   export PYTHONPATH=/path/to/slime:$PYTHONPATH
   ```

### 调试技巧

1. **测试工具调用**
   ```bash
   python -c "
   import asyncio
   from tool_sandbox import tool_registry

   async def test():
       result = await tool_registry.execute_tool(
           'code_interpreter',
           {'code': 'print(2+2)'}
       )
       print(result)

   asyncio.run(test())
   "
   ```

2. **验证数据格式**
   ```bash
   python generate_with_tools.py  # 运行测试用例
   ```

3. **检查奖励计算**
   ```python
   # 在 reward_func 中添加调试输出
   print(f"Prediction: {normalized_prediction}")
   print(f"Ground truth: {ground_truth}")
   print(f"Tool calls: {tool_call_count}")
   ```

## 🚀 性能优化

### 系统配置建议
- **CPU**: 8核以上
- **内存**: 16GB以上
- **GPU**: 支持模型并行的配置

### 训练优化
```bash
# 调整批次大小
--global-batch-size 64
--max-tokens-per-gpu 16384

# 优化并行策略
--tensor-model-parallel-size 4
--pipeline-model-parallel-size 2
```

### 工具执行优化
```python
# 减少工具调用开销
TOOL_CONFIGS["tool_concurrency"] = 32  # 根据系统能力调整
TOOL_CONFIGS["max_tool_calls"] = 6     # 限制调用次数
```

## 📚 扩展应用

### 支持其他任务类型
1. **数学问题求解**
2. **科学事实验证**
3. **逻辑推理问题**
4. **数据分析任务**

### 添加新工具
```python
# 在 ToolRegistry._register_default_tools() 中添加
self.register_tool(
    "calculator",
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform basic arithmetic operations",
            # ...
        }
    }
)
```

## 📄 参考资料

- [slime官方文档](https://thudm.github.io/slime/)
- [工具调用最佳实践](../../retool/README.md)
- [强化学习配置指南](../../docs/en/get_started/usage.md)

---

**注意**: 这是一个高级功能，需要充分的系统资源。建议先在小规模数据上测试，确认配置正确后再进行大规模训练。