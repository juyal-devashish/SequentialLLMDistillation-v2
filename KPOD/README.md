# KPOD: Keypoint-based Progressive Chain-of-Thought Distillation

Complete implementation of the KPOD paper for distilling reasoning capabilities from large language models to smaller models.

## 📋 Overview

KPOD addresses two key challenges in chain-of-thought distillation:
1. **Token Weighting**: Identifies and emphasizes keypoint tokens in rationales
2. **Progressive Distillation**: Trains models from easy to hard using curriculum learning

## 🚀 Quick Start

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Test setup
python test_setup.py
```

### Full Pipeline
```bash
# Set API key
export OPENAI_API_KEY="your-key-here"

# Run complete pipeline
python main.py --mode full_pipeline --dataset gsm8k --model google/flan-t5-large
```

## 📊 Step-by-Step Usage

### Step 1: Generate Rationales
```bash
python main.py --mode generate --dataset gsm8k
```

Generates step-by-step rationales using GPT-3.5-Turbo for training data.

### Step 2: Train Token Weighting Module
```bash
python main.py --mode train_token_weighting --dataset gsm8k
```

Trains the module to identify keypoint tokens in rationales.

### Step 3: Calculate Step Difficulties
```bash
python main.py --mode calculate_difficulties --dataset gsm8k
```

Calculates difficulty scores for each reasoning step.

### Step 4: Cluster Questions
```bash
python main.py --mode cluster --dataset gsm8k
```

Clusters questions for diversity-aware selection.

### Step 5: Train KPOD Model
```bash
python main.py --mode train --dataset gsm8k --model google/flan-t5-large
```

Trains the student model with full KPOD framework.

### Step 6: Evaluate
```bash
python main.py --mode evaluate --dataset gsm8k --model_path ./checkpoints/kpod_best_model
```

Evaluates the trained model on test set.

## 📁 Project Structure
```
KPOD/
├── configs/              # Configuration files
├── data/                 # Dataset loaders and rationale generation
├── models/               # Student model and token weighting
├── training/             # Training loops and schedulers
├── evaluation/           # Evaluation metrics
├── checkpoints/          # Saved models
├── outputs/              # Results and logs
└── main.py              # Main entry point
```

## 🎯 Expected Results

### GSM8K (Math Reasoning)

| Model | Method | Accuracy |
|-------|--------|----------|
| Flan-T5-Large (760M) | Baseline | ~18% |
| Flan-T5-Large (760M) | **KPOD** | **~22%** |
| Flan-T5-XL (3B) | Baseline | ~24% |
| Flan-T5-XL (3B) | **KPOD** | **~25%** |
| LLaMA-7B | Baseline | ~42% |
| LLaMA-7B | **KPOD** | **~47%** |

### CommonsenseQA

| Model | Method | Accuracy |
|-------|--------|----------|
| Flan-T5-Large (760M) | Baseline | ~78% |
| Flan-T5-Large (760M) | **KPOD** | **~81%** |

## ⚙️ Configuration

Edit `configs/config.py` to customize:
```python
# Model settings
student_model_name = "google/flan-t5-large"
lora_r = 128
lora_alpha = 16

# Training
batch_size = 4
epochs = 100  # For Flan-T5
learning_rate = 5e-5

# Token weighting
alpha = 0.5  # Balance between Lp and Lm

# Progressive distillation
p = 0.5       # Growth rate
c0_ratio = 0.3  # Initial difficulty
beta = 12.0    # Diversity weight
```

## 🔧 Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in config
- Use smaller model (flan-t5-small for testing)
- Enable gradient accumulation

### Slow Training
- Use fewer training samples initially
- Reduce number of epochs
- Use GPU if available

### API Rate Limits
- Reduce `batch_delay` in rationale generation
- Generate rationales in batches
- Use checkpointing (automatic)

## 📖 Citation
```bibtex
@inproceedings{feng2024kpod,
  title={Keypoint-based Progressive Chain-of-Thought Distillation for LLMs},
  author={Feng, Kaituo and Li, Changsheng and Zhang, Xiaolu and Zhou, Jun and Yuan, Ye and Wang, Guoren},
  booktitle={ICML},
  year={2024}
}
```

## 📝 License

MIT License - see LICENSE file for details.