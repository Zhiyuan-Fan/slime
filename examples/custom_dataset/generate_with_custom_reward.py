"""
自定义生成和奖励函数，用于处理true/false判断任务

支持:
1. \\boxed{} 格式的答案提取
2. true/false 答案与标签的比较
3. 详细的奖励计算逻辑
"""

import re
import json
from typing import Any, Dict

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample


def extract_boxed_answer(text: str) -> str:
    """
    从文本中提取\\boxed{}格式的答案

    Args:
        text: 包含答案的文本

    Returns:
        提取的答案内容，如果没找到返回空字符串
    """
    # 匹配\\boxed{...}格式，支持嵌套大括号
    boxed_pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    matches = re.findall(boxed_pattern, text, re.DOTALL)

    if matches:
        # 返回最后一个匹配的答案
        return matches[-1].strip().lower()

    return ""


def normalize_answer(answer: str) -> str:
    """
    标准化答案格式

    Args:
        answer: 原始答案字符串

    Returns:
        标准化后的答案 ("true", "false", 或空字符串)
    """
    answer = answer.lower().strip()

    # 直接匹配
    if answer in ["true", "false"]:
        return answer

    # 匹配常见变体
    true_variants = ["yes", "correct", "right", "1", "正确", "是", "对"]
    false_variants = ["no", "incorrect", "wrong", "0", "错误", "否", "错"]

    if answer in true_variants:
        return "true"
    elif answer in false_variants:
        return "false"

    return ""


async def generate(args, sample: Sample, sampling_params) -> Sample:
    """
    自定义生成函数，处理true/false判断任务

    Args:
        args: 训练参数
        sample: 样本数据
        sampling_params: 采样参数

    Returns:
        处理后的样本
    """
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # 构建提示词
    system_prompt = (
        "你是一个逻辑推理专家。请仔细分析给定的命题或问题，"
        "给出详细的推理过程，然后在\\boxed{}中给出最终答案（true或false）。"
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

    except Exception as e:
        print(f"生成过程中发生错误: {str(e)}")
        sample.status = Sample.Status.ABORTED
        sample.response = ""

    return sample


async def reward_func(args, sample: Sample, **kwargs) -> Dict[str, Any]:
    """
    自定义奖励函数，用于true/false判断任务

    Args:
        args: 训练参数
        sample: 样本数据

    Returns:
        包含奖励信息的字典
    """
    if not isinstance(sample, Sample):
        raise TypeError("Sample必须是Sample类的实例")

    # 获取模型的完整回答（包括提示和回复）
    full_response = sample.prompt + sample.response
    ground_truth = sample.label.lower() if sample.label else ""

    # 从回答中提取boxed答案
    predicted_answer = extract_boxed_answer(full_response)
    normalized_prediction = normalize_answer(predicted_answer)

    # 计算基础奖励
    if normalized_prediction == "":
        # 没有找到有效答案
        base_reward = -1.0
        accuracy = 0.0
        explanation = "未找到\\boxed{}格式的答案"
    elif normalized_prediction == ground_truth:
        # 答案正确
        base_reward = 1.0
        accuracy = 1.0
        explanation = f"答案正确: {normalized_prediction}"
    else:
        # 答案错误
        base_reward = -0.5
        accuracy = 0.0
        explanation = f"答案错误: 预测{normalized_prediction}，标准答案{ground_truth}"

    # 计算额外奖励因子
    response_length = len(sample.response) if sample.response else 0

    # 长度惩罚：过短或过长的回答会被轻微惩罚
    length_penalty = 0.0
    if response_length < 20:  # 回答太短
        length_penalty = -0.1
    elif response_length > 1000:  # 回答太长
        length_penalty = -0.1

    # 格式奖励：正确使用\\boxed{}格式
    format_bonus = 0.1 if "\\boxed{" in full_response else -0.2

    # 最终奖励
    final_reward = base_reward + length_penalty + format_bonus
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
        "response_length": response_length,
        "explanation": explanation,
        "metadata": {
            "task_type": "true_false_reasoning",
            "has_boxed_format": "\\boxed{" in full_response,
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

    metrics = {
        "accuracy": sum(accuracies) / len(accuracies),
        "mean_reward": sum(scores) / len(scores),
        "format_compliance": sum(format_correct) / len(format_correct),
        "num_samples": len(rewards)
    }

    return metrics


# 示例使用和测试函数
if __name__ == "__main__":
    # 测试答案提取函数
    test_cases = [
        "经过分析，这个命题是正确的。\\boxed{true}",
        "根据数学原理，答案是\\boxed{false}。",
        "我的答案是 \\boxed{True}",
        "没有boxed格式的答案",
        "\\boxed{yes}这应该被识别为true"
    ]

    print("测试答案提取功能:")
    for i, text in enumerate(test_cases, 1):
        extracted = extract_boxed_answer(text)
        normalized = normalize_answer(extracted)
        print(f"测试 {i}: '{text}'")
        print(f"  提取答案: '{extracted}'")
        print(f"  标准化后: '{normalized}'")
        print()