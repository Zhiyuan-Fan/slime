"""
不使用工具调用的生成和奖励函数，用于处理true/false判断任务

对比版本：纯语言推理 vs 工具辅助推理
"""

import re
from typing import Any, Dict

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample


def extract_boxed_answer(text: str) -> str:
    """
    Extract answer from \boxed{} format (mainstream model format)

    Args:
        text: Text containing the answer

    Returns:
        Extracted answer content, empty string if not found
    """
    # Match \boxed{...} format with nested braces support (single backslash)
    boxed_pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    matches = re.findall(boxed_pattern, text, re.DOTALL)

    if matches:
        # 返回最后一个匹配的答案
        return matches[-1].strip().lower()

    return ""


def normalize_answer(answer: str) -> str:
    """
    Normalize answer format

    Args:
        answer: Raw answer string

    Returns:
        Normalized answer ("true", "false", or empty string)
    """
    answer = answer.lower().strip()

    # 直接匹配
    if answer in ["true", "false"]:
        return answer

    # 匹配常见变体
    true_variants = ["yes", "correct", "right", "1", "正确", "是", "对", "成立"]
    false_variants = ["no", "incorrect", "wrong", "0", "错误", "否", "错", "不成立"]

    if answer in true_variants:
        return "true"
    elif answer in false_variants:
        return "false"

    return ""


async def generate(args, sample: Sample, sampling_params) -> Sample:
    """
    Generation function without tool calling (pure reasoning)

    Args:
        args: Training arguments
        sample: Sample data
        sampling_params: Sampling parameters

    Returns:
        Processed sample
    """
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # Build prompt - emphasize pure reasoning
    system_prompt = (
        "You are a logical reasoning expert. Please carefully analyze the given proposition or problem. "
        "Use your knowledge and logical reasoning ability to provide detailed analysis process, "
        "then give your final answer (true or false) in \\boxed{} format. "
        "Do not rely on external tools, solve the problem completely based on your reasoning ability."
    )

    # 格式化完整提示
    full_prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{sample.prompt}<|im_end|>
<|im_start|>assistant
"""

    # 构建请求载荷
    payload = {
        "text": full_prompt,
        "sampling_params": sampling_params,
    }

    # 发送生成请求
    try:
        output = await post(url, payload)

        # 处理不同的完成原因
        if output["meta_info"]["finish_reason"]["type"] == "abort":
            sample.status = Sample.Status.ABORTED
            return sample
        elif output["meta_info"]["finish_reason"]["type"] == "length":
            sample.status = Sample.Status.TRUNCATED
        else:
            sample.status = Sample.Status.COMPLETED

        response = output["text"]

        # 设置样本属性
        prompt_tokens = state.tokenizer(full_prompt, add_special_tokens=False)["input_ids"]
        response_tokens = state.tokenizer(response, add_special_tokens=False)["input_ids"]

        sample.tokens = prompt_tokens + response_tokens
        sample.response_length = len(response_tokens)
        sample.response = response

        # 为强化学习设置损失掩码（只对response部分计算损失）
        sample.loss_mask = [0] * len(prompt_tokens) + [1] * len(response_tokens)

        # 记录元数据
        sample.tool_call_count = 0  # 明确标记无工具调用
        sample.payload_text = full_prompt + response
        sample.payload_has_tools = False

    except Exception as e:
        print(f"生成过程中发生错误: {str(e)}")
        sample.status = Sample.Status.ABORTED
        sample.response = ""

    return sample


async def reward_func(args, sample: Sample, **kwargs) -> Dict[str, Any]:
    """
    Reward function without tool calling (pure reasoning)

    Args:
        args: Training arguments
        sample: Sample data

    Returns:
        Dictionary containing reward information
    """
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class")

    # 获取模型的完整回答
    full_response = sample.prompt + sample.response
    ground_truth = sample.label.lower() if sample.label else ""

    # 从回答中提取boxed答案
    predicted_answer = extract_boxed_answer(full_response)
    normalized_prediction = normalize_answer(predicted_answer)

    # Calculate basic accuracy reward
    if normalized_prediction == "":
        # No valid answer found
        base_reward = -1.0
        accuracy = 0.0
        explanation = "No \\boxed{} format answer found"
    elif normalized_prediction == ground_truth:
        # Correct answer
        base_reward = 1.0
        accuracy = 1.0
        explanation = f"Correct answer: {normalized_prediction}"
    else:
        # Wrong answer
        base_reward = -0.5
        accuracy = 0.0
        explanation = f"Wrong answer: predicted {normalized_prediction}, ground truth {ground_truth}"

    # 响应长度因子
    response_length = len(sample.response) if sample.response else 0
    length_penalty = 0.0
    if response_length < 50:  # 纯推理需要更多解释，最小长度要求更高
        length_penalty = -0.2
    elif response_length > 2000:  # 回答太长
        length_penalty = -0.1

    # Format bonus
    format_bonus = 0.2 if "\\boxed{" in full_response else -0.3

    # Reasoning quality bonus (more important since no tool assistance)
    reasoning_indicators = [
        "because", "therefore", "first", "second", "thus", "we can conclude", "according to", "analysis", "reasoning",
        "consider", "obviously", "hence", "in summary", "conclusion", "proof", "assume", "conversely", "since"
    ]
    reasoning_count = sum(1 for indicator in reasoning_indicators if indicator in full_response)
    reasoning_bonus = min(0.3, reasoning_count * 0.05)  # 最多0.3分的推理奖励

    # Logic structure bonus
    logic_patterns = [
        r"if.*then",      # Hypothetical reasoning
        r"when.*we",      # Conditional reasoning
        r"for.*case",     # Scope limitation
        r"example.*such", # Exemplification
        r"counterexample.*proves", # Proof by contradiction
    ]
    logic_bonus = 0.0
    for pattern in logic_patterns:
        if re.search(pattern, full_response):
            logic_bonus += 0.05
    logic_bonus = min(0.2, logic_bonus)

    # Knowledge application bonus (detect domain knowledge usage)
    knowledge_keywords = [
        "theorem", "axiom", "definition", "property", "rule", "principle", "concept", "law",
        "mathematics", "logic", "geometry", "algebra", "statistics", "probability", "physics", "chemistry"
    ]
    knowledge_bonus = 0.1 if any(keyword in full_response for keyword in knowledge_keywords) else 0.0

    # 计算最终奖励
    final_reward = (
        base_reward +
        length_penalty +
        format_bonus +
        reasoning_bonus +
        logic_bonus +
        knowledge_bonus
    )
    final_reward = max(-2.0, min(2.0, final_reward))  # 限制奖励范围

    # 返回详细的奖励信息
    result = {
        "score": final_reward,
        "pred": normalized_prediction,
        "label": ground_truth,
        "raw_prediction": predicted_answer,
        "accuracy": accuracy,
        "base_reward": base_reward,
        "length_penalty": length_penalty,
        "format_bonus": format_bonus,
        "reasoning_bonus": reasoning_bonus,
        "logic_bonus": logic_bonus,
        "knowledge_bonus": knowledge_bonus,
        "response_length": response_length,
        "reasoning_indicators_count": reasoning_count,
        "explanation": explanation,
        "metadata": {
            "task_type": "true_false_reasoning_no_tools",
            "has_boxed_format": "\\boxed{" in full_response,
            "has_tool_calls": False,
            "has_reasoning": reasoning_bonus > 0,
            "has_logic_structure": logic_bonus > 0,
            "has_knowledge": knowledge_bonus > 0,
            "extracted_answer": predicted_answer,
        }
    }

    return result


def compute_batch_metrics(rewards: list) -> Dict[str, float]:
    """
    计算批次级别的评估指标

    Args:
        rewards: 奖励列表

    Returns:
        包含各种指标的字典
    """
    if not rewards:
        return {}

    accuracies = [r.get("accuracy", 0.0) for r in rewards]
    scores = [r.get("score", 0.0) for r in rewards]
    format_correct = [r.get("metadata", {}).get("has_boxed_format", False) for r in rewards]
    has_reasoning = [r.get("metadata", {}).get("has_reasoning", False) for r in rewards]
    has_logic = [r.get("metadata", {}).get("has_logic_structure", False) for r in rewards]
    has_knowledge = [r.get("metadata", {}).get("has_knowledge", False) for r in rewards]

    metrics = {
        "accuracy": sum(accuracies) / len(accuracies),
        "mean_reward": sum(scores) / len(scores),
        "format_compliance": sum(format_correct) / len(format_correct),
        "reasoning_rate": sum(has_reasoning) / len(has_reasoning),
        "logic_structure_rate": sum(has_logic) / len(has_logic),
        "knowledge_usage_rate": sum(has_knowledge) / len(has_knowledge),
        "avg_reasoning_indicators": sum([r.get("reasoning_indicators_count", 0) for r in rewards]) / len(rewards),
        "tool_usage_rate": 0.0,  # 明确标记为0
        "avg_tool_calls": 0.0,   # 明确标记为0
        "num_samples": len(rewards)
    }

    return metrics


# 示例使用和测试函数
if __name__ == "__main__":
    # 测试答案提取功能
    test_cases = [
        "经过逻辑分析，这个命题是正确的。\\boxed{true}",
        "根据数学原理，答案是\\boxed{false}。",
        "我的推理过程：首先考虑定义，其次分析性质，因此\\boxed{True}",
        "没有boxed格式的答案",
        "综合考虑各种因素，\\boxed{正确}这是我的结论"
    ]

    print("测试纯推理答案提取功能:")
    for i, text in enumerate(test_cases, 1):
        extracted = extract_boxed_answer(text)
        normalized = normalize_answer(extracted)
        print(f"测试 {i}: '{text}'")
        print(f"  提取答案: '{extracted}'")
        print(f"  标准化后: '{normalized}'")
        print()

    # 测试推理质量评估
    reasoning_text = """
    首先，我们需要分析什么是质数。根据数学定义，质数是大于1的自然数，
    只能被1和它本身整除。其次，考虑偶数的定义：能被2整除的整数。

    现在让我们推理：如果所有质数都是奇数，那么不存在偶质数。
    但是，我们知道2是质数（只能被1和2整除），同时2也是偶数。
    因此，存在一个反例证明原命题是错误的。

    综上所述，\\boxed{false}
    """

    print("测试推理质量评估:")
    reasoning_indicators = [
        "因为", "所以", "首先", "其次", "因此", "可以得出", "根据", "分析", "推理",
        "考虑", "显然", "由此", "综上", "总结", "结论", "证明", "假设", "反之"
    ]

    count = sum(1 for indicator in reasoning_indicators if indicator in reasoning_text)
    print(f"推理指示词数量: {count}")
    print(f"推理奖励: {min(0.3, count * 0.05)}")

    # 检测逻辑结构
    logic_patterns = [
        r"如果.*那么",
        r"当.*时",
        r"对于.*来说",
        r"例如.*这样",
        r"反例.*证明",
    ]

    logic_count = sum(1 for pattern in logic_patterns if re.search(pattern, reasoning_text))
    print(f"逻辑结构数量: {logic_count}")
    print(f"逻辑奖励: {min(0.2, logic_count * 0.05)}")