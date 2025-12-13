import os
import argparse
from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F
from datasets import load_dataset
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

torch.cuda.empty_cache()     # frees *cached* memory
torch.cuda.synchronize()
torch.cuda.ipc_collect()
torch.cuda.reset_peak_memory_stats()
torch.cuda.reset_accumulated_memory_stats()
torch.cuda.set_per_process_memory_fraction(0.95, device=0)
print(torch.cuda.memory_allocated()/1024**2, "MB allocated")
print(torch.cuda.memory_reserved()/1024**2, "MB reserved")


def build_config(
    teacher: str,
    student: str,
    output_dir: str,
    dataset_name: str = "mlabonne/FineTome-100k",
    dataset_split: str = "train",
    num_samples: Optional[int] = 100000,
    seed: int = 42,
    max_length: int = 512,
    num_train_epochs: int = 5,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    save_steps: int = 1000,
    logging_steps: int = 1,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.05,
    warmup_ratio: float = 0.1,
    lr_scheduler_type: str = "cosine",
    resume_from_checkpoint: Optional[str] = None,
    fp16: bool = False,
    bf16: bool = True,
    temperature: float = 2.0,
    alpha: float = 0.5,
    use_flash_attention: bool = False,
) -> Dict[str, Any]:
    """
    Build a config dict compatible with the original distil_logits.py script,
    but parameterized so it can be reused for sequential and parallel flows.
    """
    project_name = "distil-logits-sequential"

    config: Dict[str, Any] = {
        "project_name": project_name,
        "dataset": {
            "name": dataset_name,
            "split": dataset_split,
            "seed": seed,
        },
        "models": {
            "teacher": teacher,
            "student": student,
        },
        "tokenizer": {
            "max_length": max_length,
            "chat_template": "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}",
        },
        "training": {
            "output_dir": output_dir,
            "num_train_epochs": num_train_epochs,
            "per_device_train_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "save_steps": save_steps,
            "logging_steps": logging_steps,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "lr_scheduler_type": lr_scheduler_type,
            "resume_from_checkpoint": resume_from_checkpoint,
            "fp16": fp16,
            "bf16": bf16,
        },
        "distillation": {
            "temperature": temperature,
            "alpha": alpha,
        },
        "model_config": {
            "use_flash_attention": use_flash_attention,
        },
    }

    if num_samples is not None:
        config["dataset"]["num_samples"] = num_samples

    return config


def prepare_dataset(config: Dict[str, Any], student_tokenizer: AutoTokenizer):
    dataset_cfg = config["dataset"]

    dataset = load_dataset(dataset_cfg["name"], split=dataset_cfg["split"])
    dataset = dataset.shuffle(seed=dataset_cfg["seed"])

    if "num_samples" in dataset_cfg and dataset_cfg["num_samples"] is not None:
        dataset = dataset.select(range(dataset_cfg["num_samples"]))

    # Apply ShareGPT-style formatting as in distil_logits.py
    def sharegpt_format(example):
        conversations = example["conversations"]
        message = []

        if isinstance(conversations, list):
            for conversation in conversations:
                if isinstance(conversation, dict):
                    if conversation.get("from") == "human":
                        message.append(
                            {"role": "user", "content": conversation.get("value", "")}
                        )
                    elif conversation.get("from") == "gpt":
                        message.append(
                            {
                                "role": "assistant",
                                "content": conversation.get("value", ""),
                            }
                        )
                    elif conversation.get("from") == "system":
                        message.insert(
                            0,
                            {
                                "role": "system",
                                "content": conversation.get("value", ""),
                            },
                        )

        if not any(msg.get("role") == "system" for msg in message):
            message.insert(
                0, {"role": "system", "content": "You are a helpful assistant."}
            )

        text = student_tokenizer.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        return {"text": text}

    print("Preprocessing and tokenizing dataset...")
    original_columns = dataset.column_names
    dataset = dataset.map(sharegpt_format, remove_columns=original_columns)

    def tokenize_function(examples):
        return student_tokenizer(
            examples["text"],
            truncation=True,
            max_length=config["tokenizer"]["max_length"],
            padding="max_length",
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=8,
        remove_columns=["text"],
    )
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)

    return tokenized_dataset


def pad_logits(student_logits: torch.Tensor, teacher_logits: torch.Tensor):
    student_size, teacher_size = student_logits.size(-1), teacher_logits.size(-1)
    if student_size != teacher_size:
        pad_size = abs(student_size - teacher_size)
        pad_tensor = torch.zeros(
            (*teacher_logits.shape[:-1], pad_size),
            dtype=teacher_logits.dtype,
            device=teacher_logits.device,
        )
        if student_size < teacher_size:
            return torch.cat([student_logits, pad_tensor], dim=-1), teacher_logits
        else:
            return student_logits, torch.cat([teacher_logits, pad_tensor], dim=-1)
    return student_logits, teacher_logits


class LogitsTrainer(SFTTrainer):
    """
    Custom trainer that performs logit-based KD, adapted from distil_logits.py
    but parameterized by a config dict instead of relying on globals.
    """

    def __init__(self, *args, distil_config: Dict[str, Any], **kwargs):
        super().__init__(*args, **kwargs)
        self.distil_config = distil_config
        self.teacher_model = None

    def compute_loss(
        self, model, inputs, return_outputs: bool = False, num_items_in_batch=None
    ):
        if self.teacher_model is None:
            raise ValueError("teacher_model must be set on the trainer before training.")

        device = next(model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        # Make sure teacher is on the same device
        self.teacher_model = self.teacher_model.to(device)

        student_model = model.module if hasattr(model, "module") else model
        teacher_model = (
            self.teacher_model.module
            if hasattr(self.teacher_model, "module")
            else self.teacher_model
        )

        student_outputs = student_model(**inputs)
        with torch.no_grad():
            teacher_outputs = teacher_model(**inputs)

        custom_loss = self.distillation_loss(
            model,
            student_outputs.logits,
            teacher_outputs.logits,
            inputs,
            student_outputs.loss,
        )
        return (custom_loss, student_outputs) if return_outputs else custom_loss

    def distillation_loss(
        self,
        model,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        inputs,
        original_loss: torch.Tensor,
    ):
        device = next(model.parameters()).device
        student_logits, teacher_logits = pad_logits(
            student_logits.to(device), teacher_logits.to(device)
        )

        temperature = self.distil_config["distillation"]["temperature"]
        alpha = self.distil_config["distillation"]["alpha"]
        max_length = self.distil_config["tokenizer"]["max_length"]

        student_logits_scaled = student_logits / temperature
        teacher_logits_scaled = teacher_logits / temperature

        loss_kd = (
            F.kl_div(
                F.log_softmax(student_logits_scaled, dim=-1),
                F.softmax(teacher_logits_scaled, dim=-1),
                reduction="batchmean",
            )
            * (temperature**2)
            / max_length
        )

        return alpha * loss_kd + (1 - alpha) * original_loss


def run_logit_distillation(config: Dict[str, Any]):
    os.environ["WANDB_PROJECT"] = config["project_name"]

    # Load tokenizers
    teacher_tokenizer = AutoTokenizer.from_pretrained(config["models"]["teacher"])
    student_tokenizer = AutoTokenizer.from_pretrained(config["models"]["student"])
    student_tokenizer.chat_template = config["tokenizer"]["chat_template"]

    # Prepare dataset
    tokenized_dataset = prepare_dataset(config, student_tokenizer)

    print("Dataset preparation complete. Loading models...")

    # Load models with configurable flash attention
    training_cfg = config["training"]
    dtype = (
        torch.bfloat16
        if training_cfg.get("bf16", False)
        else torch.float16
        if training_cfg.get("fp16", False)
        else torch.float32
    )

    model_kwargs: Dict[str, Any] = {"torch_dtype": dtype}
    if config["model_config"]["use_flash_attention"]:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    teacher_model = AutoModelForCausalLM.from_pretrained(
        config["models"]["teacher"], **model_kwargs
    )
    student_model = AutoModelForCausalLM.from_pretrained(
        config["models"]["student"], **model_kwargs
    )

    os.makedirs(training_cfg["output_dir"], exist_ok=True)

    # Training arguments and trainer
    training_arguments = TrainingArguments(**training_cfg)

    trainer = LogitsTrainer(
        model=student_model,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        args=training_arguments,
        distil_config=config,
    )

    # Attach the teacher model
    trainer.teacher_model = teacher_model

    # Train
    trainer.train(resume_from_checkpoint=training_cfg.get("resume_from_checkpoint"))

    # Save the final model (student)
    trainer.save_model(training_cfg["output_dir"])

    # Explicitly clean up to ease GPU memory pressure if this is called repeatedly
    del teacher_model, student_model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Single-step logit-based distillation between a teacher and a student.\n"
            "Designed to be reusable for sequential and parallel distillation chains."
        )
    )

    parser.add_argument(
        "--teacher",
        type=str,
        required=True,
        help="HF model ID or local path for the teacher model (e.g. arcee-ai/Arcee-Spark).",
    )
    parser.add_argument(
        "--student",
        type=str,
        required=True,
        help="HF model ID for the student model (e.g. Qwen/Qwen3-1.7B).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where the distilled student checkpoint will be saved.",
    )

    # Dataset
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="mlabonne/FineTome-100k",
        help="Dataset name or path (default: mlabonne/FineTome-100k).",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        help="Dataset split to use (default: train).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Optional number of samples to subsample from the dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset shuffling.",
    )

    # Tokenizer / sequence
    parser.add_argument(
        "--max-length",
        type=int,
        default=4096,
        help="Maximum sequence length for tokenization and training.",
    )

    # Training hyperparameters
    parser.add_argument(
        "--num-train-epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=1,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=1000,
        help="Save checkpoint every N steps.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=1,
        help="Log training metrics every N steps.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.05,
        help="Weight decay.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup ratio for the learning rate scheduler.",
    )
    parser.add_argument(
        "--lr-scheduler-type",
        type=str,
        default="cosine",
        help="Learning rate scheduler type (e.g. cosine, linear).",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (if any).",
    )

    # Precision / attention
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 training.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use BF16 training.",
    )
    parser.add_argument(
        "--no-flash-attention",
        action="store_true",
        help="Disable FlashAttention-2 even if available.",
    )

    # Distillation-specific
    parser.add_argument(
        "--temperature",
        type=float,
        default=2.0,
        help="Distillation temperature.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Interpolation between KD loss and original loss (0-1).",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    torch.cuda.empty_cache()

    use_flash_attention = not args.no_flash_attention

    config = build_config(
        teacher=args.teacher,
        student=args.student,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        num_samples=args.num_samples,
        seed=args.seed,
        max_length=args.max_length,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        resume_from_checkpoint=args.resume_from_checkpoint,
        fp16=args.fp16,
        bf16=args.bf16,
        temperature=args.temperature,
        alpha=args.alpha,
        use_flash_attention=use_flash_attention,
    )

    run_logit_distillation(config)


if __name__ == "__main__":
    main()


