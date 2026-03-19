from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from datasets import Dataset, DatasetDict

from src.entity.config_entity import ModelConfig, TrainingConfig
from src.training.trainer_runner import _build_training_args, run_training
from src.utils.exception import ModelError


@contextmanager
def _dummy_run():
    yield


@pytest.fixture
def training_config(tmp_path: Path) -> TrainingConfig:
    return TrainingConfig(
        root_dir=tmp_path / "artifacts" / "training",
        dataset_path=tmp_path / "artifacts" / "data",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        logging_steps=5,
        save_steps=10,
        eval_steps=10,
        warmup_ratio=0.05,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )


@pytest.fixture
def model_config(tmp_path: Path) -> ModelConfig:
    return ModelConfig(
        root_dir=tmp_path / "artifacts" / "model",
        model_name="mock-model/Mock-LLM-8B",
        cache_dir=tmp_path / "cache",
        use_4bit=True,
    )


def test_build_training_args(training_config: TrainingConfig) -> None:
    args = _build_training_args(training_config)

    assert args.output_dir == str(training_config.root_dir)
    assert args.num_train_epochs == training_config.num_train_epochs
    assert args.per_device_train_batch_size == training_config.per_device_train_batch_size
    assert args.learning_rate == training_config.learning_rate
    assert args.logging_steps == training_config.logging_steps


def test_run_training_missing_dataset(
    training_config: TrainingConfig, model_config: ModelConfig
) -> None:
    model = MagicMock()

    with pytest.raises(ModelError, match="Dataset"):
        run_training(model, training_config, model_config)


def test_run_training_success(
    training_config: TrainingConfig, model_config: ModelConfig, tmp_path: Path
) -> None:
    dataset = Dataset.from_dict(
        {"input_ids": [[1, 2]], "attention_mask": [[1, 1]]}
    )
    dataset_dict = DatasetDict({"train": dataset, "validation": dataset})
    dataset_path = tmp_path / "data"
    dataset_dict.save_to_disk(str(dataset_path))

    config = TrainingConfig(
        root_dir=training_config.root_dir,
        dataset_path=dataset_path,
        num_train_epochs=training_config.num_train_epochs,
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        per_device_eval_batch_size=training_config.per_device_eval_batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        learning_rate=training_config.learning_rate,
        logging_steps=training_config.logging_steps,
        save_steps=training_config.save_steps,
        eval_steps=training_config.eval_steps,
        warmup_ratio=training_config.warmup_ratio,
        lora_r=training_config.lora_r,
        lora_alpha=training_config.lora_alpha,
        lora_dropout=training_config.lora_dropout,
        target_modules=training_config.target_modules,
    )

    mock_trainer = MagicMock()
    with patch("src.training.trainer_runner.SFTTrainer", return_value=mock_trainer), patch(
        "src.training.trainer_runner.mlflow"
    ) as mock_mlflow:
        mock_mlflow.start_run.return_value = _dummy_run()
        trainer = run_training(MagicMock(), config, model_config)

    mock_trainer.train.assert_called_once()
    assert trainer == mock_trainer


def test_run_training_oom_raises_model_error(
    training_config: TrainingConfig, model_config: ModelConfig, tmp_path: Path
) -> None:
    dataset = Dataset.from_dict(
        {"input_ids": [[1, 2]], "attention_mask": [[1, 1]]}
    )
    dataset_dict = DatasetDict({"train": dataset, "validation": dataset})
    dataset_path = tmp_path / "data"
    dataset_dict.save_to_disk(str(dataset_path))

    config = TrainingConfig(
        root_dir=training_config.root_dir,
        dataset_path=dataset_path,
        num_train_epochs=training_config.num_train_epochs,
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        per_device_eval_batch_size=training_config.per_device_eval_batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        learning_rate=training_config.learning_rate,
        logging_steps=training_config.logging_steps,
        save_steps=training_config.save_steps,
        eval_steps=training_config.eval_steps,
        warmup_ratio=training_config.warmup_ratio,
        lora_r=training_config.lora_r,
        lora_alpha=training_config.lora_alpha,
        lora_dropout=training_config.lora_dropout,
        target_modules=training_config.target_modules,
    )

    mock_trainer = MagicMock()
    mock_trainer.train.side_effect = torch.cuda.OutOfMemoryError("CUDA OOM")

    with patch("src.training.trainer_runner.SFTTrainer", return_value=mock_trainer), patch(
        "src.training.trainer_runner.mlflow"
    ) as mock_mlflow:
        mock_mlflow.start_run.return_value = _dummy_run()
        with pytest.raises(ModelError, match="OOM"):
            run_training(MagicMock(), config, model_config)
