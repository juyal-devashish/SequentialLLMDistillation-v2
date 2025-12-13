import os
import argparse
from typing import Optional

from distil_logits_sequential import (
    build_config as build_logits_config,
    run_logit_distillation,
)

# Hidden-state sequential distillation is currently disabled.
# from distil_hidden_sequential import build_config as build_hidden_config, run_hidden_distillation


DEFAULT_TEACHER = "arcee-ai/Arcee-Spark"
DEFAULT_STUDENT_A = "Qwen/Qwen2-1.5B"
DEFAULT_STUDENT_B = "Qwen/Qwen2-0.5B"


def sequential_logits_chain(
    base_output_dir: str,
    teacher: str = DEFAULT_TEACHER,
    student_a: str = DEFAULT_STUDENT_A,
    student_b: str = DEFAULT_STUDENT_B,
    num_samples: Optional[int] = None,
    max_length: int = 1024,
):
    """
    Run the sequential logit-based chain:
    Arcee-Spark → Qwen2-1.5B → Qwen2-0.5B
    """
    os.makedirs(base_output_dir, exist_ok=True)

    step1_dir = os.path.join(base_output_dir, "qwen2-1_5b_from_arcee_logits")
    step2_dir = os.path.join(base_output_dir, "qwen2-0_5b_from_qwen2-1_5b_logits")

    # Step 1: teacher → student A
    cfg1 = build_logits_config(
        teacher=teacher,
        student=student_a,
        output_dir=step1_dir,
        num_samples=num_samples,
        max_length=max_length,
    )
    run_logit_distillation(cfg1)

    # Step 2: student A checkpoint as new teacher → student B
    cfg2 = build_logits_config(
        teacher=step1_dir,
        student=student_b,
        output_dir=step2_dir,
        num_samples=num_samples,
        max_length=max_length,
    )
    run_logit_distillation(cfg2)


# def sequential_hidden_chain(
#     base_output_dir: str,
#     teacher: str = DEFAULT_TEACHER,
#     student_a: str = DEFAULT_STUDENT_A,
#     student_b: str = DEFAULT_STUDENT_B,
# ):
#     """
#     Run the sequential hidden-state-based chain:
#     Arcee-Spark → Qwen3-1.7B → Qwen3-0.6B
#     """
#     os.makedirs(base_output_dir, exist_ok=True)
#
#     step1_dir = os.path.join(base_output_dir, "qwen3-1_7b_from_arcee_hidden")
#     step2_dir = os.path.join(base_output_dir, "qwen3-0_6b_from_qwen3-1_7b_hidden")
#
#     # Step 1: teacher → student A
#     cfg1 = build_hidden_config(
#         teacher=teacher,
#         student=student_a,
#         output_dir=step1_dir,
#     )
#     run_hidden_distillation(cfg1)
#
#     # Step 2: student A checkpoint as new teacher → student B
#     cfg2 = build_hidden_config(
#         teacher=step1_dir,
#         student=student_b,
#         output_dir=step2_dir,
#     )
#     run_hidden_distillation(cfg2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convenience runner for sequential Arcee-Spark → Qwen2-1.5B → Qwen2-0.5B "
            "distillation chain (logits only; hidden-states disabled)."
        )
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["logits"],
        default="logits",
        help="Which sequential chain to run (only logits is currently enabled).",
    )
    parser.add_argument(
        "--base-output-dir",
        type=str,
        default="./sequential_checkpoints",
        help="Base directory under which step checkpoints will be stored.",
    )
    parser.add_argument(
        "--teacher",
        type=str,
        default=DEFAULT_TEACHER,
        help=f"Teacher model ID (default: {DEFAULT_TEACHER}).",
    )
    parser.add_argument(
        "--student-a",
        type=str,
        default=DEFAULT_STUDENT_A,
        help=f"First student model ID (default: {DEFAULT_STUDENT_A}).",
    )
    parser.add_argument(
        "--student-b",
        type=str,
        default=DEFAULT_STUDENT_B,
        help=f"Second student model ID (default: {DEFAULT_STUDENT_B}).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Optional number of samples to use from the dataset (shared across steps).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=4096,
        help="Maximum sequence length for tokenization and training (shared across steps).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Only logits mode is enabled for now.
    sequential_logits_chain(
        base_output_dir=os.path.join(args.base_output_dir, "logits"),
        teacher=args.teacher,
        student_a=args.student_a,
        student_b=args.student_b,
        num_samples=args.num_samples,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()


