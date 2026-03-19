from pathlib import Path
from typing import Any

import mlflow
import torch
from datasets import load_from_disk
from transformers import TrainingArguments
from transformers.utils import is_torch_bf16_gpu_available
from trl import SFTTrainer

from src.entity.config_entity import ModelConfig, TrainingConfig
from src.training.mlflow_callback import MLflowProgressCallback
from src.utils.exception import ModelError
from src.utils.logger import logger


def _build_training_args(config: TrainingConfig) -> TrainingArguments:
    """
    Constructs HuggingFace TrainingArguments from TrainingConfig.

    Args:
        config: TrainingConfig with all hyperparameters.

    Returns:
        A fully configured TrainingArguments instance.
    """
    logger.debug(
        f"Building TrainingArguments — epochs={config.num_train_epochs}, "
        f"batch={config.per_device_train_batch_size}, lr={config.learning_rate}, "
        f"grad_accum={config.gradient_accumulation_steps}"
    )
    # Only enable bf16 when the current environment supports it.
    use_bf16 = torch.cuda.is_available() and is_torch_bf16_gpu_available()

    return TrainingArguments(
        output_dir=str(config.root_dir),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        bf16=use_bf16,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="none",
    )


def _log_mlflow_params(config: TrainingConfig, model_config: ModelConfig) -> None:
    """Logs training hyperparameters to the active MLflow run."""
    mlflow.log_params(
        {
            "model_name": model_config.model_name,
            "use_4bit": model_config.use_4bit,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "num_train_epochs": config.num_train_epochs,
            "per_device_train_batch_size": config.per_device_train_batch_size,
            "learning_rate": config.learning_rate,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
        }
    )
    logger.debug("MLflow params logged.")


def run_training(
    model: Any,
    config: TrainingConfig,
    model_config: ModelConfig,
) -> SFTTrainer:
    """
    Loads the tokenized dataset and runs SFTTrainer inside an MLflow tracking run.

    Args:
        model: The PEFT-wrapped model ready for training.
        config: TrainingConfig with dataset path and training hyperparameters.
        model_config: ModelConfig for MLflow parameter logging.

    Returns:
        The SFTTrainer instance after training completes.

    Raises:
        ModelError: On CUDA OOM or any unexpected training failure.
    """
    logger.info(f"Loading tokenized dataset from: '{config.dataset_path}'")
    try:
        dataset_dict = load_from_disk(str(config.dataset_path))
    except FileNotFoundError as e:
        logger.error(
            f"Dataset not found at '{config.dataset_path}'. "
            "Run the data transformation step first."
        )
        raise ModelError("Dataset directory not found. Cannot start training.") from e
    except Exception as e:
        logger.error(f"Failed to load dataset from disk: {e}")
        raise ModelError("Unexpected error while loading dataset.") from e

    logger.info(
        f"Dataset loaded — Train: {len(dataset_dict['train']):,}, "
        f"Validation: {len(dataset_dict['validation']):,}"
    )

    training_args = _build_training_args(config)

    mlflow.set_experiment("llama-finetune")
    with mlflow.start_run():
        _log_mlflow_params(config, model_config)

        logger.info("Starting SFT training...")
        try:
            trainer = SFTTrainer(  # type: ignore[call-arg]
                model=model,
                args=training_args,
                train_dataset=dataset_dict["train"],  # type: ignore[arg-type]
                eval_dataset=dataset_dict["validation"],  # type: ignore[arg-type]
                callbacks=[MLflowProgressCallback()],
            )
            trainer.train()
        except torch.cuda.OutOfMemoryError as e:
            logger.error(
                "CUDA Out of Memory during training. "
                "Try reducing 'per_device_train_batch_size' or increasing "
                "'gradient_accumulation_steps' in your training config."
            )
            raise ModelError("OOM error during SFTTrainer.train().") from e
        except Exception as e:
            logger.error(f"Unexpected error during SFT training: {e}")
            raise ModelError("Unexpected failure during model training.") from e

    logger.info("Training complete.")
    return trainer
