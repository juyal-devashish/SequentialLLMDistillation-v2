# Explain Like I’m a 500M-Parameter Model: Sequential LLM Knowledge Distillation

A novel framework for compressing large language models through progressive distillation with intermediate teacher assistants.

## Overview

KPOD addresses a fundamental challenge in knowledge distillation: **when the capacity gap between teacher and student is too large, direct distillation fails**. We propose progressive distillation through intermediate-sized models, decomposing a 14× compression into manageable steps.

### Key Results

| Model | Parameters | Perplexity ↓ | Top-1 Acc ↑ | KL Div ↓ |
|-------|-----------|--------------|-------------|----------|
| Teacher (Arcee-Spark) | 7.0B | 2.89 | 75.68% | - |
| **Direct 0.5B** | 0.5B | 2.28 | 78.30% | 6522.56 |
| **Sequential 0.5B (KPOD)** | **0.5B** | **1.76** | **85.05%** | **6072.96** |

**KPOD achieves 22.8% lower perplexity and 8.6% higher accuracy compared to direct distillation.**

## Architecture

```
Sequential Distillation:
Teacher (7B) → Intermediate (1.5B) → Student (0.5B)
   [Stage 1: 4.7× compression] → [Stage 2: 3× compression]

Parallel Distillation:
Teacher (7B) ─┬→ Intermediate (1.5B) → Student (0.5B)
              └────────────────────────→ (co-training)
```

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/KPOD.git
cd KPOD

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- PyTorch 2.0+
- transformers
- datasets
- trl
- accelerate

## Quick Start

### Sequential Distillation

```python
from kpod import SequentialDistillation

# Initialize distillation pipeline
distiller = SequentialDistillation(
    teacher_model="arcee-ai/Arcee-Spark",
    intermediate_model="Qwen/Qwen2-1.5B",
    student_model="Qwen/Qwen2-0.5B",
    dataset="mlabonne/FineTome-100k",
    temperature=2.0,
    alpha=0.5  # distillation coefficient
)

# Stage 1: Teacher → Intermediate
distiller.train_stage1(
    output_dir="./results/intermediate",
    num_epochs=3,
    learning_rate=2e-5
)

# Stage 2: Intermediate → Student
distiller.train_stage2(
    output_dir="./results/student",
    num_epochs=3,
    learning_rate=2e-5
)
```

### Parallel Distillation

```python
from kpod import ParallelDistillation

distiller = ParallelDistillation(
    teacher_model="arcee-ai/Arcee-Spark",
    intermediate_model="Qwen/Qwen2-1.5B",
    student_model="Qwen/Qwen2-0.5B",
    dataset="mlabonne/FineTome-100k"
)

distiller.train(
    output_dir="./results/parallel",
    num_epochs=3
)
```

## Evaluation

```python
from kpod import evaluate_model

results = evaluate_model(
    model_path="./results/student",
    test_dataset="mlabonne/FineTome-100k",
    metrics=["perplexity", "top_k_accuracy", "kl_divergence"]
)

print(results)
```

## Key Features

- ✅ **Progressive Distillation**: Gradual capacity reduction through intermediate models
- ✅ **Sequential & Parallel Variants**: Choose based on quality vs. speed trade-off
- ✅ **Keypoint Extraction**: Emphasize critical reasoning steps (KL-based)
- ✅ **Memory Efficient**: Gradient accumulation, mixed-precision training
- ✅ **HPC Ready**: Optimized for cluster deployment with automatic cleanup
- ✅ **Comprehensive Evaluation**: Perplexity, Top-k accuracy, KL divergence, qualitative analysis

## Configuration

Edit `configs/default_config.yaml`:

```yaml
distillation:
  temperature: 2.0
  alpha: 0.5  # KL vs CE weight
  beta: 0.1   # keypoint weight

training:
  batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 2e-5
  num_epochs: 3
  warmup_ratio: 0.1
  weight_decay: 0.05

models:
  teacher: "arcee-ai/Arcee-Spark"
  intermediate: "Qwen/Qwen2-1.5B"
  student: "Qwen/Qwen2-0.5B"
```

## Results

### Quantitative Performance

- **22.8% perplexity reduction** vs. direct distillation
- **8.6% Top-1 accuracy improvement** (85.05% vs. 78.30%)
- **6.9% lower KL divergence** (better teacher approximation)

### Qualitative Analysis

The distilled 0.5B model successfully:
- Generates coherent, technically accurate responses
- Maintains structured formatting (lists, code blocks, sections)
- Produces functionally correct code implementations
- Preserves explanation strategies and terminology

## Methodology

### Why Progressive Distillation Works

Direct distillation struggles with large capacity gaps (14× in our case) because the student lacks sufficient representational capacity. KPOD solves this by:

1. **Decomposing compression**: 14× → (4.7× + 3×) manageable steps
2. **Stable intermediate teachers**: Each stage learns from fully converged teacher
3. **Knowledge refinement**: Sequential process acts as distillation + refinement

### Comparison to Baseline Methods

| Method | Teacher→Student | Perplexity | Top-1 Acc | Training Time |
|--------|----------------|------------|-----------|---------------|
| Direct Distillation | 7B → 0.5B | 2.28 | 78.30% | 1× |
| Sequential KPOD | 7B → 1.5B → 0.5B | **1.76** | **85.05%** | 2× |
| Parallel KPOD | 7B → [1.5B, 0.5B] | 1.81 | 84.12% | 1.2× |

## Hardware Requirements

- **Minimum**: 1× GPU with 24GB VRAM (e.g., RTX 3090, A5000)
- **Recommended**: 1× GPU with 40GB+ VRAM (e.g., A100)
- **CPU RAM**: 32GB+
- **Storage**: ~50GB for models and checkpoints

## Acknowledgments

- Built on top of [DistillKit](https://github.com/arcee-ai/DistillKit) by Arcee AI
- Uses models from [Qwen2](https://huggingface.co/Qwen) and [Arcee-Spark](https://huggingface.co/arcee-ai/Arcee-Spark)
- Dataset: [FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k) by mlabonne
