"""
支持多轮工具调用的生成和奖励函数，用于处理true/false判断任务

功能特性:
1. 多轮工具调用：模型可以多次调用Python代码执行工具
2. \\boxed{} 格式答案提取
3. 工具使用情况的奖励计算
4. 详细的推理过程跟踪
"""

import re
import json
import asyncio
from typing import Any, Dict

try:
    from jinja2 import Template
except ImportError:
    raise ImportError("Jinja2 is required. Please install it with: pip install jinja2")

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

# Import tool sandbox functionality
from tool_sandbox import SEMAPHORE, TOOL_CONFIGS, tool_registry

# Jinja2 template for tool-enabled conversations
TOOL_TEMPLATE = """<|im_start|>system
{%- if messages[0]['role'] == 'system' %}
{{- messages[0]['content'] }}
{%- else %}
You are a helpful reasoning assistant.
{%- endif %}
{%- if tools %}
# Tools

You may call one or more functions to assist with your analysis and reasoning.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{%- for tool in tools %}
{{- tool | tojson }}
{%- endfor %}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

After using tools to gather information or perform calculations, provide your final answer in \\boxed{true} or \\boxed{false} format.
{%- endif %}
<|im_end|>
{%- for message in messages %}
{%- if message['role'] == 'user' %}
<|im_start|>user
{{- message['content'] }}<|im_end|>
{%- elif message['role'] == 'assistant' %}
<|im_start|>assistant
{{- message['content'] }}<|im_end|>
{%- endif %}
{%- endfor %}
<|im_start|>assistant
"""


def format_conversation_with_tools(
    prompt: str,
    tools: list = None,
    system_prompt: str = None,
    messages: list = None
) -> str:
    """Format conversation using Jinja2 template with tool support"""
    template = Template(TOOL_TEMPLATE)

    # Prepare messages
    messages_to_render = []

    # Add system message
    if system_prompt:
        system_content = system_prompt
    else:
        system_content = (
            "You are a logical reasoning expert. Please carefully analyze the given proposition or problem. "
            "You can use code execution tools for computation, verification, or analysis. "
            "Then provide your final answer in \\boxed{} format (true or false)."
        )

    messages_to_render.append({"role": "system", "content": system_content})

    # Add user message
    if prompt:
        messages_to_render.append({"role": "user", "content": prompt})

    # Add assistant responses from previous turns if provided
    if messages:
        messages_to_render.extend(messages)

    # Render template
    formatted_text = template.render(messages=messages_to_render, tools=tools or [])
    return formatted_text


def extract_boxed_answer(text: str) -> str:
    """
    从文本中提取\\boxed{}格式的答案 (主流模型格式)

    Args:
        text: 包含答案的文本

    Returns:
        提取的答案内容，如果没找到返回空字符串
    """
    # 匹配\boxed{...}格式，支持嵌套大括号 (主流格式用单反斜杠)
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


def postprocess_predictions(prediction: str) -> tuple[str, str]:
    """Extract action and content from prediction string"""
    # Check for Answer: \boxed{...} format (highest priority)
    answer_pattern = r"Answer:\s*\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}"
    answer_match = re.search(answer_pattern, prediction, re.DOTALL)
    if answer_match:
        content = answer_match.group(1).strip()
        return "answer", content

    # Check for standalone \boxed{...} format (主流模型格式)
    boxed_pattern = r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}"
    boxed_match = re.search(boxed_pattern, prediction, re.DOTALL)
    if boxed_match:
        content = boxed_match.group(1).strip()
        return "answer", content

    # Check for <tool_call> tags
    tool_call_pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
    tool_call_match = re.search(tool_call_pattern, prediction, re.DOTALL)
    if tool_call_match:
        try:
            # Clean up the JSON string
            json_str = tool_call_match.group(1)
            json_str = json_str.replace("\n", "\\n")
            tool_call_data = json.loads(json_str)
            tool_name = tool_call_data.get("name")
            arguments = tool_call_data.get("arguments", {})

            if tool_name == "code_interpreter":
                code = arguments.get("code", "")
                if code.strip():
                    return "code", code
        except (json.JSONDecodeError, KeyError, AttributeError):
            pass

    return None, ""


def postprocess_responses(resp: str) -> str:
    """Post-process response to ensure tag completeness"""
    # Handle Answer: \\boxed{...} format (highest priority)
    if "Answer:" in resp and "\\boxed{" in resp:
        answer_pattern = r"Answer:\s*\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}"
        matches = list(re.finditer(answer_pattern, resp, re.DOTALL))
        if matches:
            last_match = matches[-1]
            return resp[:last_match.end()]

    # Handle standalone \\boxed{...} format
    if "\\boxed{" in resp:
        boxed_pattern = r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}"
        matches = list(re.finditer(boxed_pattern, resp, re.DOTALL))
        if matches:
            last_match = matches[-1]
            return resp[:last_match.end()]

    # Handle <tool_call> tags
    if "<tool_call>" in resp:
        tool_call_pattern = r"<tool_call>\s*\{.*?\}\s*</tool_call>"
        matches = list(re.finditer(tool_call_pattern, resp, re.DOTALL))
        if matches:
            last_match = matches[-1]
            return resp[:last_match.end()]

    return resp


async def execute_predictions(prediction: str) -> tuple[str, bool]:
    """Execute predictions and return results"""
    action, content = postprocess_predictions(prediction)

    if action == "code":
        # Execute Python code
        code = content.strip()
        if code:
            async with SEMAPHORE:
                result = await tool_registry.execute_tool("code_interpreter", {"code": code})
            next_obs = f"\n\n<interpreter>\n{result}\n</interpreter>\n\n"
            done = False
        else:
            next_obs = "\n\n<interpreter>\nError: No Python code found\n</interpreter>\n\n"
            done = False
    elif action == "answer":
        # Final answer provided
        next_obs = ""
        done = True
    else:
        # Invalid action, provide guidance
        next_obs = (
            "\nI need to provide a clear action. "
            "If I want to execute code to assist reasoning, I should use <tool_call> format to call code_interpreter tool. "
            "If I want to give a final answer, I should use \\boxed{true} or \\boxed{false} format.\n"
        )
        done = False

    return next_obs, done


async def generate(args, sample: Sample, sampling_params) -> Sample:
    """
    自定义生成函数，支持多轮工具调用的true/false判断任务

    Args:
        args: 训练参数
        sample: 样本数据
        sampling_params: 采样参数

    Returns:
        处理后的样本
    """
    assert not args.partial_rollout, "Partial rollout is not supported for this function at the moment."

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # 设置初始提示和工具
    tool_specs = tool_registry.get_tool_specs()
    prompt = format_conversation_with_tools(prompt=sample.prompt, tools=tool_specs)

    prompt_tokens_ids = state.tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response = ""
    response_token_ids = []
    loss_masks = []
    tool_call_count = 0  # 跟踪实际工具调用轮数

    for turn in range(TOOL_CONFIGS["max_turns"]):
        # 构建请求载荷
        payload = {
            "text": prompt + response,
            "sampling_params": sampling_params,
        }

        # 记录调试信息到wandb
        try:
            import wandb
            if wandb.run is not None:
                available_tools = len(tool_specs)
                tools_used = response.count("<interpreter>")

                wandb.log({
                    "debug/payload_length": len(prompt + response),
                    "debug/available_tools": available_tools,
                    "debug/tools_used": tools_used,
                    "debug/turn": turn,
                    "debug/tool_call_count": tool_call_count,
                })
        except ImportError:
            pass

        # 发送生成请求
        output = await post(url, payload)

        # 处理中止情况
        if output["meta_info"]["finish_reason"]["type"] == "abort":
            sample.status = Sample.Status.ABORTED
            return sample

        cur_response = output["text"]
        cur_response = postprocess_responses(cur_response)

        # 记录当前回应的token
        cur_response_token_ids = state.tokenizer(cur_response, add_special_tokens=False)["input_ids"]
        response += cur_response
        response_token_ids += cur_response_token_ids
        loss_masks += [1] * len(cur_response_token_ids)

        # 检查长度限制
        if output["meta_info"]["finish_reason"]["type"] == "length":
            break

        # 执行预测并获取下一步观察
        next_obs, done = await execute_predictions(cur_response)

        if done:
            break

        # 统计工具调用次数
        if "<interpreter>" in next_obs:
            tool_call_count += 1

        assert next_obs != "", "Next observation should not be empty."
        obs_tokens_ids = state.tokenizer(next_obs, add_special_tokens=False)["input_ids"]
        response += next_obs
        response_token_ids += obs_tokens_ids
        loss_masks += [0] * len(obs_tokens_ids)  # 工具输出不参与损失计算

        # 检查是否达到最大工具调用次数
        if tool_call_count >= TOOL_CONFIGS["max_tool_calls"]:
            break

    # 设置样本属性
    sample.tokens = prompt_tokens_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_mask = loss_masks

    # 存储有用的元数据
    sample.tool_call_count = tool_call_count
    sample.payload_text = prompt + response
    sample.payload_has_system = "<|im_start|>system" in prompt + response
    sample.payload_has_tools = "# Tools" in prompt + response

    # 设置状态
    match output["meta_info"]["finish_reason"]["type"]:
        case "length":
            sample.status = Sample.Status.TRUNCATED
        case "abort":
            sample.status = Sample.Status.ABORTED
        case "stop":
            sample.status = Sample.Status.COMPLETED

    return sample


async def reward_func(args, sample: Sample, **kwargs) -> Dict[str, Any]:
    """
    支持工具调用的自定义奖励函数

    Args:
        args: 训练参数
        sample: 样本数据

    Returns:
        包含奖励信息的字典
    """
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    # 获取模型的完整回答
    full_response = sample.prompt + sample.response
    ground_truth = sample.label.lower() if sample.label else ""

    # 从回答中提取boxed答案
    predicted_answer = extract_boxed_answer(full_response)
    normalized_prediction = normalize_answer(predicted_answer)

    # 获取工具调用次数
    tool_call_count = getattr(sample, "tool_call_count", 0)

    # 计算基础准确性奖励
    if normalized_prediction == "":
        # 没有找到有效答案
        base_reward = -1.0
        accuracy = 0.0
        explanation = "No \\boxed{} format answer found"
    elif normalized_prediction == ground_truth:
        # 答案正确
        base_reward = 1.0
        accuracy = 1.0
        explanation = f"Correct answer: {normalized_prediction}"
    else:
        # 答案错误
        base_reward = -0.5
        accuracy = 0.0
        explanation = f"Wrong answer: predicted {normalized_prediction}, ground truth {ground_truth}"

    # 工具使用奖励/惩罚
    tool_reward = 0.0
    if tool_call_count > 0:
        if accuracy == 1.0:
            # 正确答案且使用了工具：鼓励合理的工具使用
            tool_reward = min(0.3, tool_call_count * 0.1)
        else:
            # 错误答案但使用了工具：轻微奖励尝试
            tool_reward = min(0.1, tool_call_count * 0.05)
    else:
        # 没有使用工具
        if accuracy == 1.0:
            # 正确答案但没使用工具：可能是简单问题，不惩罚
            tool_reward = 0.0
        else:
            # 错误答案且没使用工具：轻微惩罚
            tool_reward = -0.1

    # 响应长度因子
    response_length = len(sample.response) if sample.response else 0
    length_penalty = 0.0
    if response_length < 30:  # 回答太短
        length_penalty = -0.1
    elif response_length > 2000:  # 回答太长
        length_penalty = -0.1

    # 格式奖励
    format_bonus = 0.1 if "\\boxed{" in full_response else -0.2

    # 推理质量奖励（检查是否有推理过程）
    reasoning_indicators = ["因为", "所以", "首先", "其次", "因此", "可以得出", "根据"]
    reasoning_bonus = 0.1 if any(indicator in full_response for indicator in reasoning_indicators) else 0.0

    # 计算最终奖励
    final_reward = base_reward + tool_reward + length_penalty + format_bonus + reasoning_bonus
    final_reward = max(-2.0, min(2.0, final_reward))  # 限制奖励范围

    # 返回详细的奖励信息
    result = {
        "score": final_reward,
        "pred": normalized_prediction,
        "label": ground_truth,
        "raw_prediction": predicted_answer,
        "accuracy": accuracy,
        "base_reward": base_reward,
        "tool_reward": tool_reward,
        "length_penalty": length_penalty,
        "format_bonus": format_bonus,
        "reasoning_bonus": reasoning_bonus,
        "response_length": response_length,
        "tool_call_count": tool_call_count,
        "explanation": explanation,
        "metadata": {
            "task_type": "true_false_reasoning_with_tools",
            "has_boxed_format": "\\boxed{" in full_response,
            "has_tool_calls": tool_call_count > 0,
            "has_reasoning": reasoning_bonus > 0,
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
    tool_counts = [r.get("tool_call_count", 0) for r in rewards]
    format_correct = [r.get("metadata", {}).get("has_boxed_format", False) for r in rewards]
    has_reasoning = [r.get("metadata", {}).get("has_reasoning", False) for r in rewards]

    metrics = {
        "accuracy": sum(accuracies) / len(accuracies),
        "mean_reward": sum(scores) / len(scores),
        "format_compliance": sum(format_correct) / len(format_correct),
        "reasoning_rate": sum(has_reasoning) / len(has_reasoning),
        "avg_tool_calls": sum(tool_counts) / len(tool_counts),
        "tool_usage_rate": len([t for t in tool_counts if t > 0]) / len(tool_counts),
        "num_samples": len(rewards)
    }

    return metrics


# 示例使用和测试函数
if __name__ == "__main__":
    # 测试答案提取功能
    test_cases = [
        "经过计算分析，这个命题是正确的。\\boxed{true}",
        "根据数学原理，答案是\\boxed{false}。",
        "我需要使用代码来验证这个问题。",
        "<tool_call>{\"name\": \"code_interpreter\", \"arguments\": {\"code\": \"print(2+2)\"}}</tool_call>",
        "Answer: \\boxed{True}这是最终答案"
    ]

    print("测试预测提取功能:")
    for i, text in enumerate(test_cases, 1):
        action, content = postprocess_predictions(text)
        print(f"测试 {i}: '{text}'")
        print(f"  动作: '{action}', 内容: '{content}'")
        print()

    # 测试答案提取
    print("测试答案提取功能:")
    for i, text in enumerate(test_cases, 1):
        extracted = extract_boxed_answer(text)
        normalized = normalize_answer(extracted)
        print(f"测试 {i}: 提取='{extracted}', 标准化='{normalized}'")