import os
import argparse
from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F
from datasets import load_dataset
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments


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
    logging_steps: int = 2,
    save_total_limit: int = 2,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.2,
    lr_scheduler_type: str = "linear",
    resume_from_checkpoint: Optional[str] = None,
    fp16: bool = False,
    bf16: bool = True,
    max_grad_norm: float = 1.0,
    group_by_length: bool = False,
    temperature: float = 2.0,
    alpha: float = 0.5,
    use_flash_attention: bool = False,
) -> Dict[str, Any]:
    """
    Build a config dict compatible with the original distil_hidden.py script,
    but parameterized so it can be reused for sequential and parallel flows.
    """
    project_name = "distil-hidden-sequential"

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
            "save_total_limit": save_total_limit,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "lr_scheduler_type": lr_scheduler_type,
            "resume_from_checkpoint": resume_from_checkpoint,
            "fp16": fp16,
            "bf16": bf16,
            "max_grad_norm": max_grad_norm,
            "group_by_length": group_by_length,
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


def prepare_dataset(
    config: Dict[str, Any],
    student_tokenizer: AutoTokenizer,
    teacher_tokenizer: AutoTokenizer,
):
    dataset_cfg = config["dataset"]

    dataset = load_dataset(dataset_cfg["name"], split=dataset_cfg["split"])
    if dataset_cfg.get("num_samples") is not None:
        dataset = dataset.select(range(dataset_cfg["num_samples"]))
    dataset = dataset.shuffle(seed=dataset_cfg["seed"])

    def prepare_example(example):
        system = "You are a helpful assistant chatbot."
        conversations = example["conversations"]

        message = [{"role": "system", "content": system}]

        for conversation in conversations:
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

        student_text = student_tokenizer.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        teacher_text = teacher_tokenizer.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )

        student_encodings = student_tokenizer(
            student_text,
            truncation=True,
            max_length=config["tokenizer"]["max_length"],
            padding="max_length",
        )
        teacher_encodings = teacher_tokenizer(
            teacher_text,
            truncation=True,
            max_length=config["tokenizer"]["max_length"],
            padding="max_length",
        )

        return {
            "input_ids": student_encodings["input_ids"],
            "attention_mask": student_encodings["attention_mask"],
            "teacher_input_ids": teacher_encodings["input_ids"],
            "teacher_attention_mask": teacher_encodings["attention_mask"],
        }

    print("Preprocessing and tokenizing dataset for hidden-state distillation...")
    original_columns = dataset.column_names
    dataset = dataset.map(prepare_example, remove_columns=original_columns)
    return dataset


class MultiLayerAdaptationLayer(torch.nn.Module):
    """
    Projects student hidden states into teacher hidden dimension and aligns layers.
    Adapted from distil_hidden.py.
    """

    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        num_student_layers: int,
        num_teacher_layers: int,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.projections = torch.nn.ModuleList(
            [
                torch.nn.Linear(student_dim, teacher_dim, dtype=dtype)
                for _ in range(num_student_layers)
            ]
        )
        self.layer_mapping = self.create_layer_mapping(
            num_student_layers, num_teacher_layers
        )
        self.dtype = dtype

    @staticmethod
    def create_layer_mapping(num_student_layers: int, num_teacher_layers: int):
        return {
            i: round(i * (num_teacher_layers - 1) / (num_student_layers - 1))
            for i in range(num_student_layers)
        }

    def forward(self, student_hidden_states):
        adapted_hidden_states = []
        for i, hidden_state in enumerate(student_hidden_states):
            if i >= len(self.projections):
                break
            adapted_hidden_states.append(self.projections[i](hidden_state.to(self.dtype)))
        return adapted_hidden_states


class HiddenStatesSFTTrainer(SFTTrainer):
    """
    Custom SFTTrainer that adds a hidden-state-based KD loss on top of the SFT loss.
    """

    def __init__(self, *args, **kwargs):
        self.distil_config = kwargs.pop("distil_config")
        self.teacher_model = kwargs.pop("teacher_model")
        self.adaptation_layer = kwargs.pop("adaptation_layer")
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs: bool = False):
        # Student forward
        student_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }

        labels = inputs["labels"]

        student_outputs = model(
            **student_inputs, labels=labels, output_hidden_states=True
        )

        original_loss = student_outputs.loss

        teacher_model = (
            self.teacher_model.module
            if hasattr(self.teacher_model, "module")
            else self.teacher_model
        )

        with torch.no_grad():
            teacher_inputs = {
                "input_ids": inputs["teacher_input_ids"],
                "attention_mask": inputs["teacher_attention_mask"],
            }

            teacher_outputs = teacher_model(
                **teacher_inputs, output_hidden_states=True
            )

        custom_loss = self.distillation_loss(
            student_outputs, teacher_outputs, inputs, original_loss
        )
        return (custom_loss, student_outputs) if return_outputs else custom_loss

    def distillation_loss(
        self, student_outputs, teacher_outputs, inputs, original_loss: torch.Tensor
    ):
        student_hidden_states = student_outputs.hidden_states
        teacher_hidden_states = teacher_outputs.hidden_states

        # Ensure adaptation layer is on the right device
        self.adaptation_layer = self.adaptation_layer.to(
            student_hidden_states[0].device
        )
        adapted_student_hidden_states = self.adaptation_layer(student_hidden_states)

        temperature = self.distil_config["distillation"]["temperature"]
        alpha = self.distil_config["distillation"]["alpha"]

        total_loss_kd = 0.0
        for student_idx, teacher_idx in self.adaptation_layer.layer_mapping.items():
            if student_idx >= len(adapted_student_hidden_states):
                continue

            student_hidden = adapted_student_hidden_states[student_idx]
            teacher_hidden = teacher_hidden_states[teacher_idx]

            if student_hidden.shape != teacher_hidden.shape:
                raise ValueError(
                    f"Shape mismatch: student {student_hidden.shape} vs teacher {teacher_hidden.shape}"
                )

            # KD on hidden states via KL-div on softened distributions
            student_logits = student_hidden / temperature
            teacher_logits = teacher_hidden / temperature

            loss_kd = F.kl_div(
                F.log_softmax(student_logits, dim=-1),
                F.softmax(teacher_logits, dim=-1),
                reduction="batchmean",
            ) * (temperature**2)

            total_loss_kd = total_loss_kd + loss_kd

        avg_loss_kd = total_loss_kd / max(
            1, len(self.adaptation_layer.layer_mapping)
        )
        hidden_dim = adapted_student_hidden_states[0].size(-1)
        scaled_loss_kd = avg_loss_kd / hidden_dim

        total_loss = alpha * scaled_loss_kd + (1 - alpha) * original_loss
        return total_loss


def run_hidden_distillation(config: Dict[str, Any]):
    os.environ["WANDB_PROJECT"] = config["project_name"]
    os.makedirs(config["training"]["output_dir"], exist_ok=True)

    # Load tokenizers
    teacher_tokenizer = AutoTokenizer.from_pretrained(config["models"]["teacher"])
    student_tokenizer = AutoTokenizer.from_pretrained(config["models"]["student"])

    student_tokenizer.chat_template = config["tokenizer"]["chat_template"]

    # Prepare dataset
    dataset = prepare_dataset(config, student_tokenizer, teacher_tokenizer)

    print("Dataset preparation complete. Loading models...")

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

    adaptation_dtype = torch.bfloat16 if training_cfg.get("bf16", False) else torch.float32
    adaptation_layer = MultiLayerAdaptationLayer(
        student_model.config.hidden_size,
        teacher_model.config.hidden_size,
        student_model.config.num_hidden_layers,
        teacher_model.config.num_hidden_layers,
        dtype=adaptation_dtype,
    )

    training_arguments = TrainingArguments(
        **training_cfg,
        remove_unused_columns=False,
    )

    trainer = HiddenStatesSFTTrainer(
        model=student_model,
        train_dataset=dataset,
        max_seq_length=config["tokenizer"]["max_length"],
        tokenizer=student_tokenizer,
        args=training_arguments,
        packing=training_cfg.get("packing", False),
        distil_config=config,
        teacher_model=teacher_model,
        adaptation_layer=adaptation_layer,
    )

    trainer.train(resume_from_checkpoint=training_cfg.get("resume_from_checkpoint"))

    # Save the final student model
    trainer.save_model(training_cfg["output_dir"])

    # Save the adaptation layer
    adaptation_layer_path = os.path.join(
        training_cfg["output_dir"], "adaptation_layer.pth"
    )
    torch.save(adaptation_layer.state_dict(), adaptation_layer_path)

    # Explicit cleanup for repeated calls
    del teacher_model, student_model, trainer, adaptation_layer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Single-step hidden-state-based distillation between a teacher and a student.\n"
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
        default=1000,
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
        default=2,
        help="Log training metrics every N steps.",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=2,
        help="Maximum number of checkpoints to keep.",
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
        default=0.01,
        help="Weight decay.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.2,
        help="Warmup ratio for the learning rate scheduler.",
    )
    parser.add_argument(
        "--lr-scheduler-type",
        type=str,
        default="linear",
        help="Learning rate scheduler type (e.g. cosine, linear).",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (if any).",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping.",
    )
    parser.add_argument(
        "--group-by-length",
        action="store_true",
        help="Group sequences of similar length together for efficiency.",
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
        save_total_limit=args.save_total_limit,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        resume_from_checkpoint=args.resume_from_checkpoint,
        fp16=args.fp16,
        bf16=args.bf16,
        max_grad_norm=args.max_grad_norm,
        group_by_length=args.group_by_length,
        temperature=args.temperature,
        alpha=args.alpha,
        use_flash_attention=use_flash_attention,
    )

    run_hidden_distillation(config)


if __name__ == "__main__":
    main()


