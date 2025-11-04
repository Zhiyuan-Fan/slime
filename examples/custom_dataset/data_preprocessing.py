"""
Data preprocessing script: Convert custom true/false judgment datasets to slime format

Input format: {"prompt": "...", "label": "true/false"}
Output format: slime standard format for training and inference
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Union

try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    print("Warning: datasets library not available. Install with: pip install datasets")
    HF_DATASETS_AVAILABLE = False


# English template for solution evaluation
SOLUTION_EVALUATION_TEMPLATE = """Given a problem and a model's solution, evaluate whether the solution is correct. Return your answer in \\boxed{{}} format as either "true" or "false".

Problem:
{problem}

Model Solution:
{solution}

Please analyze the solution step by step and determine if it correctly solves the given problem. Provide your final judgment in \\boxed{{true}} or \\boxed{{false}} format.
"""

# Alternative template for mathematical reasoning
MATH_REASONING_TEMPLATE = """Evaluate the correctness of the following mathematical solution. Analyze the reasoning process and conclude with \\boxed{{true}} if correct or \\boxed{{false}} if incorrect.

Problem Statement:
{problem}

Proposed Solution:
{solution}

Your evaluation:"""


def convert_from_hf_dataset(
    dataset_name: str,
    output_path: str,
    split: str = 'train',
    max_samples: Optional[int] = None,
    template_type: str = 'solution_eval',
    problem_key: str = 'problem',
    solution_key: str = 'answer',
    correct_key: str = 'correct'
):
    """
    Convert Hugging Face dataset to slime format for solution evaluation

    Args:
        dataset_name: Name of the HF dataset
        output_path: Output JSONL file path
        split: Dataset split to use ('train', 'test', 'validation')
        max_samples: Maximum number of samples to process
        template_type: Type of template ('solution_eval' or 'math_reasoning')
        problem_key: Key for problem text in dataset
        solution_key: Key for solution text in dataset
        correct_key: Key for correctness label in dataset
    """
    if not HF_DATASETS_AVAILABLE:
        raise ImportError("datasets library is required for HF dataset conversion")

    print(f"Loading dataset: {dataset_name} (split: {split})")
    dataset = load_dataset(dataset_name, split=split)

    # Select template
    template = SOLUTION_EVALUATION_TEMPLATE if template_type == 'solution_eval' else MATH_REASONING_TEMPLATE

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    converted_count = 0
    skipped_count = 0

    # Limit samples if specified
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    with open(output_path, 'w', encoding='utf-8') as outfile:
        for idx, item in enumerate(dataset):
            try:
                # Extract fields
                problem = item.get(problem_key, "")
                solution_raw = item.get(solution_key, "")
                is_correct = item.get(correct_key, None)

                # Validate required fields
                if not problem or not solution_raw or is_correct is None:
                    print(f"Warning: Missing required fields in item {idx}, skipping")
                    skipped_count += 1
                    continue

                # Clean solution (remove thinking tags if present)
                if "</think>" in solution_raw:
                    solution = solution_raw.split("</think>")[1].strip()
                else:
                    solution = solution_raw.strip()

                # Format prompt using template
                prompt = template.format(
                    problem=problem.strip(),
                    solution=solution
                )

                # Convert label
                label = 'true' if is_correct else 'false'

                # Create slime format sample
                converted_sample = {
                    "prompt": prompt,
                    "label": label,
                    "metadata": {
                        "source_dataset": dataset_name,
                        "source_index": idx,
                        "task_type": "solution_evaluation",
                        "template_type": template_type,
                        "original_problem": problem,
                        "original_solution": solution
                    }
                }

                # Write to output file
                outfile.write(json.dumps(converted_sample, ensure_ascii=False) + '\n')
                converted_count += 1

                if (converted_count + 1) % 1000 == 0:
                    print(f"Processed {converted_count + 1} samples...")

            except Exception as e:
                print(f"Error processing item {idx}: {str(e)}, skipping")
                skipped_count += 1
                continue

    print(f"HF dataset conversion completed:")
    print(f"  - Converted: {converted_count} samples")
    print(f"  - Skipped: {skipped_count} samples")
    print(f"  - Output file: {output_path}")

    return converted_count



def convert_custom_dataset(input_path: str, output_path: str):
    """
    将自定义数据集转换为slime格式

    Args:
        input_path: 输入JSONL文件路径
        output_path: 输出JSONL文件路径
    """

    # 确保输出目录存在
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    converted_count = 0

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:

        for line_no, line in enumerate(infile, 1):
            try:
                # 解析原始数据
                data = json.loads(line.strip())

                # 验证必需字段
                if 'prompt' not in data or 'label' not in data:
                    print(f"警告: 第{line_no}行缺少必需字段，跳过")
                    continue

                # 验证label格式
                if data['label'].lower() not in ['true', 'false']:
                    print(f"警告: 第{line_no}行label格式错误: {data['label']}，跳过")
                    continue

                # 转换为slime格式
                converted_sample = {
                    "prompt": data["prompt"],
                    "label": data["label"].lower(),  # 标准化为小写
                    "metadata": {
                        "source_line": line_no,
                        "task_type": "true_false_reasoning"
                    }
                }

                # 写入输出文件
                outfile.write(json.dumps(converted_sample, ensure_ascii=False) + '\n')
                converted_count += 1

            except json.JSONDecodeError:
                print(f"错误: 第{line_no}行JSON格式错误，跳过")
                continue
            except Exception as e:
                print(f"错误: 第{line_no}行处理失败: {str(e)}，跳过")
                continue

    print(f"数据预处理完成: 成功转换 {converted_count} 条数据")
    print(f"输出文件: {output_path}")


def split_dataset(
    input_path: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    shuffle: bool = True,
    seed: int = 42
) -> Dict[str, str]:
    """
    Split dataset into training, validation, and test sets

    Args:
        input_path: Input JSONL file path
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        shuffle: Whether to shuffle data before splitting
        seed: Random seed for reproducible shuffling

    Returns:
        Dictionary with paths to split files
    """
    print(f"Splitting dataset: {input_path}")

    # Read all data
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total_count = len(lines)
    print(f"Total samples: {total_count}")

    # Shuffle data if requested
    if shuffle:
        random.seed(seed)
        random.shuffle(lines)
        print(f"Data shuffled with seed: {seed}")

    # Calculate split sizes
    train_count = int(total_count * train_ratio)
    val_count = int(total_count * val_ratio)
    test_count = total_count - train_count - val_count

    # Get file path information
    input_path = Path(input_path)
    base_name = input_path.stem
    output_dir = input_path.parent

    # Split data
    train_data = lines[:train_count]
    val_data = lines[train_count:train_count + val_count]
    test_data = lines[train_count + val_count:]

    # Save split data
    splits = [
        (train_data, f"{base_name}_train.jsonl", "Training"),
        (val_data, f"{base_name}_val.jsonl", "Validation"),
        (test_data, f"{base_name}_test.jsonl", "Test")
    ]

    result_paths = {}

    for data, filename, split_name in splits:
        output_path = output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(data)

        result_paths[split_name.lower()] = str(output_path)
        print(f"{split_name} set: {len(data)} samples -> {output_path}")

    # Print summary
    print(f"\nDataset split summary:")
    print(f"  Training: {len(train_data)} samples ({train_count/total_count*100:.1f}%)")
    print(f"  Validation: {len(val_data)} samples ({val_count/total_count*100:.1f}%)")
    print(f"  Test: {len(test_data)} samples ({test_count/total_count*100:.1f}%)")

    return result_paths


def validate_dataset(input_path: str) -> Dict[str, Union[int, float, List]]:
    """
    Validate dataset format and provide statistics

    Args:
        input_path: Path to JSONL dataset file

    Returns:
        Dictionary with validation results and statistics
    """
    print(f"Validating dataset: {input_path}")

    stats = {
        "total_samples": 0,
        "valid_samples": 0,
        "invalid_samples": 0,
        "label_distribution": {"true": 0, "false": 0},
        "avg_prompt_length": 0,
        "errors": []
    }

    prompt_lengths = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            stats["total_samples"] += 1

            try:
                data = json.loads(line.strip())

                # Check required fields
                if 'prompt' not in data or 'label' not in data:
                    stats["invalid_samples"] += 1
                    stats["errors"].append(f"Line {line_no}: Missing required fields")
                    continue

                # Validate label
                label = data['label'].lower()
                if label not in ['true', 'false']:
                    stats["invalid_samples"] += 1
                    stats["errors"].append(f"Line {line_no}: Invalid label '{data['label']}'")
                    continue

                # Count valid sample
                stats["valid_samples"] += 1
                stats["label_distribution"][label] += 1
                prompt_lengths.append(len(data['prompt']))

            except json.JSONDecodeError:
                stats["invalid_samples"] += 1
                stats["errors"].append(f"Line {line_no}: JSON decode error")
            except Exception as e:
                stats["invalid_samples"] += 1
                stats["errors"].append(f"Line {line_no}: {str(e)}")

    # Calculate statistics
    if prompt_lengths:
        stats["avg_prompt_length"] = sum(prompt_lengths) / len(prompt_lengths)
        stats["min_prompt_length"] = min(prompt_lengths)
        stats["max_prompt_length"] = max(prompt_lengths)

    # Print validation results
    print(f"\nValidation Results:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Valid samples: {stats['valid_samples']}")
    print(f"  Invalid samples: {stats['invalid_samples']}")
    print(f"  Success rate: {stats['valid_samples']/stats['total_samples']*100:.1f}%")

    if stats['valid_samples'] > 0:
        print(f"\nLabel Distribution:")
        total_valid = stats['valid_samples']
        for label, count in stats['label_distribution'].items():
            percentage = count / total_valid * 100
            print(f"  {label}: {count} samples ({percentage:.1f}%)")

        print(f"\nPrompt Length Statistics:")
        print(f"  Average: {stats['avg_prompt_length']:.1f} characters")
        print(f"  Range: {stats['min_prompt_length']} - {stats['max_prompt_length']} characters")

    if stats['errors']:
        print(f"\nErrors (showing first 5):")
        for error in stats['errors'][:5]:
            print(f"  - {error}")
        if len(stats['errors']) > 5:
            print(f"  ... and {len(stats['errors']) - 5} more errors")

    return stats


def create_sample_dataset(
    output_path: str,
    num_samples: int = 100,
    task_types: List[str] = None
) -> str:
    """
    Create a sample dataset for testing and development

    Args:
        output_path: Output file path
        num_samples: Number of samples to generate
        task_types: Types of reasoning tasks to include

    Returns:
        Path to created dataset
    """
    if task_types is None:
        task_types = ['math', 'logic', 'science', 'general']

    print(f"Creating sample dataset with {num_samples} samples...")

    # Sample templates for different task types
    sample_templates = {
        'math': [
            ("Is the following statement true? All prime numbers greater than 2 are odd. Answer in \\boxed{{}} format.", "true"),
            ("Evaluate: Is 1 considered a prime number in modern mathematics? Answer in \\boxed{{}} format.", "false"),
            ("True or false: The sum of two even numbers is always even. Answer in \\boxed{{}} format.", "true"),
            ("Is this correct? The square root of 16 is both 4 and -4. Answer in \\boxed{{}} format.", "false")
        ],
        'logic': [
            ("Logical reasoning: If all cats are mammals and all mammals are animals, then all cats are animals. Is this reasoning valid? Answer in \\boxed{{}} format.", "true"),
            ("Consider this statement: If it rains, then the ground gets wet. The ground is wet, therefore it rained. Is this logical reasoning sound? Answer in \\boxed{{}} format.", "false"),
            ("True or false: The statement 'This sentence is false' creates a logical paradox. Answer in \\boxed{{}} format.", "true")
        ],
        'science': [
            ("Physics question: Does light always travel in straight lines? Answer in \\boxed{{}} format.", "false"),
            ("Chemistry: Is water (H2O) composed of two hydrogen atoms and one oxygen atom? Answer in \\boxed{{}} format.", "true"),
            ("Biology: Are all bacteria harmful to humans? Answer in \\boxed{{}} format.", "false")
        ],
        'general': [
            ("Geography: Is the Sahara Desert located in Africa? Answer in \\boxed{{}} format.", "true"),
            ("History: Did World War II end in 1945? Answer in \\boxed{{}} format.", "true"),
            ("Literature: Is Shakespeare considered the author of 'Romeo and Juliet'? Answer in \\boxed{{}} format.", "true")
        ]
    }

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    samples_per_type = num_samples // len(task_types)
    generated_count = 0

    with open(output_path, 'w', encoding='utf-8') as f:
        for task_type in task_types:
            templates = sample_templates.get(task_type, sample_templates['general'])

            for i in range(samples_per_type):
                # Cycle through templates
                template_idx = i % len(templates)
                prompt, label = templates[template_idx]

                sample = {
                    "prompt": prompt,
                    "label": label,
                    "metadata": {
                        "task_type": f"sample_{task_type}",
                        "generated": True,
                        "sample_id": generated_count
                    }
                }

                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                generated_count += 1

    print(f"Created sample dataset: {output_path}")
    print(f"Generated {generated_count} samples across {len(task_types)} task types")

    return output_path


def main():
    """Main function with command line interface"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Data preprocessing for True/False reasoning tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create sample dataset
  python data_preprocessing.py --create-sample --num-samples 500

  # Convert HuggingFace dataset
  python data_preprocessing.py --hf-dataset "your/dataset" --output "converted.jsonl"

  # Convert custom dataset and split
  python data_preprocessing.py --input "raw.jsonl" --output "converted.jsonl" --split

  # Validate existing dataset
  python data_preprocessing.py --validate "dataset.jsonl"
        """
    )

    # Action arguments
    parser.add_argument("--create-sample", action="store_true",
                       help="Create a sample dataset for testing")
    parser.add_argument("--hf-dataset", type=str,
                       help="Hugging Face dataset name to convert")
    parser.add_argument("--input", type=str,
                       help="Input JSONL file path for custom conversion")
    parser.add_argument("--validate", type=str,
                       help="Validate dataset at given path")

    # Output arguments
    parser.add_argument("--output", type=str, default="./data/custom_dataset/converted_dataset.jsonl",
                       help="Output file path")

    # Processing options
    parser.add_argument("--split", action="store_true",
                       help="Split dataset into train/val/test")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                       help="Training set ratio for splitting")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                       help="Validation set ratio for splitting")
    parser.add_argument("--shuffle", action="store_true", default=True,
                       help="Shuffle data before splitting")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducible operations")

    # Sample creation options
    parser.add_argument("--num-samples", type=int, default=200,
                       help="Number of samples to create")
    parser.add_argument("--task-types", nargs="+", default=['math', 'logic', 'science', 'general'],
                       help="Task types for sample creation")

    # HF dataset options
    parser.add_argument("--hf-split", type=str, default="train",
                       help="HuggingFace dataset split to use")
    parser.add_argument("--max-samples", type=int,
                       help="Maximum samples to process from HF dataset")
    parser.add_argument("--template-type", choices=['solution_eval', 'math_reasoning'],
                       default='solution_eval', help="Template type for HF conversion")

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    try:
        # Create sample dataset
        if args.create_sample:
            print("=" * 50)
            print("Creating Sample Dataset")
            print("=" * 50)

            created_path = create_sample_dataset(
                output_path=args.output,
                num_samples=args.num_samples,
                task_types=args.task_types
            )

            if args.split:
                print("\nSplitting sample dataset...")
                split_paths = split_dataset(
                    created_path,
                    train_ratio=args.train_ratio,
                    val_ratio=args.val_ratio,
                    shuffle=args.shuffle,
                    seed=args.seed
                )

        # Convert HuggingFace dataset
        elif args.hf_dataset:
            print("=" * 50)
            print(f"Converting HuggingFace Dataset: {args.hf_dataset}")
            print("=" * 50)

            convert_from_hf_dataset(
                dataset_name=args.hf_dataset,
                output_path=args.output,
                split=args.hf_split,
                max_samples=args.max_samples,
                template_type=args.template_type
            )

            if args.split:
                print("\nSplitting converted dataset...")
                split_paths = split_dataset(
                    args.output,
                    train_ratio=args.train_ratio,
                    val_ratio=args.val_ratio,
                    shuffle=args.shuffle,
                    seed=args.seed
                )

        # Convert custom dataset
        elif args.input:
            print("=" * 50)
            print(f"Converting Custom Dataset: {args.input}")
            print("=" * 50)

            if not os.path.exists(args.input):
                print(f"Error: Input file not found: {args.input}")
                return

            convert_custom_dataset(args.input, args.output)

            if args.split:
                print("\nSplitting converted dataset...")
                split_paths = split_dataset(
                    args.output,
                    train_ratio=args.train_ratio,
                    val_ratio=args.val_ratio,
                    shuffle=args.shuffle,
                    seed=args.seed
                )

        # Validate dataset
        elif args.validate:
            print("=" * 50)
            print(f"Validating Dataset: {args.validate}")
            print("=" * 50)

            if not os.path.exists(args.validate):
                print(f"Error: Validation file not found: {args.validate}")
                return

            stats = validate_dataset(args.validate)

        # Default: show help and create sample
        else:
            print("=" * 50)
            print("No specific action specified. Creating sample dataset...")
            print("=" * 50)

            sample_path = "./data/custom_dataset/sample_dataset.jsonl"
            created_path = create_sample_dataset(
                output_path=sample_path,
                num_samples=100,
                task_types=['math', 'logic', 'science', 'general']
            )

            print(f"\n✅ Sample dataset created: {sample_path}")
            print(f"📝 To use your own data, run:")
            print(f"   python data_preprocessing.py --input your_data.jsonl --output converted.jsonl --split")

            # Validate the sample dataset
            print(f"\n🔍 Validating sample dataset...")
            validate_dataset(sample_path)

    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()