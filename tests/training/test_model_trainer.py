from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.entity.config_entity import ModelConfig, TrainingConfig


@pytest.fixture
def training_config():
    return TrainingConfig(
        root_dir=Path("/mock/artifacts/training"),
        dataset_path=Path("/mock/artifacts/data_transformation"),
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
def model_config():
    return ModelConfig(
        root_dir=Path("/mock/artifacts/model"),
        model_name="mock-model/Mock-LLM-8B",
        cache_dir=Path("/mock/cache/models"),
        use_4bit=True,
    )


@patch("src.training.model_trainer.AutoModelForCausalLM.from_pretrained")
@patch("src.training.model_trainer.os.makedirs")
def test_init_creates_directories_and_loads_model(
    mock_makedirs, mock_from_pretrained, training_config, model_config
):
    """ModelTrainer.__init__ creates root_dir and loads model via _load_model."""
    mock_model = MagicMock()
    mock_from_pretrained.return_value = mock_model

    from src.training.model_trainer import ModelTrainer

    trainer = ModelTrainer(config=training_config, model_config=model_config)

    mock_makedirs.assert_any_call(model_config.cache_dir, exist_ok=True)
    mock_makedirs.assert_any_call(model_config.root_dir, exist_ok=True)
    mock_from_pretrained.assert_called_once()
    assert trainer.model == mock_model


@patch("src.training.model_trainer.get_peft_model")
@patch("src.training.model_trainer.AutoModelForCausalLM.from_pretrained")
@patch("src.training.model_trainer.os.makedirs")
def test_setup_peft_model_uses_correct_lora_config(
    mock_makedirs, mock_from_pretrained, mock_get_peft, training_config, model_config
):
    """setup_peft_model builds the right LoraConfig and wraps the model."""
    mock_model = MagicMock()
    mock_model.get_nb_trainable_parameters.return_value = (1_000, 8_000_000_000)
    mock_from_pretrained.return_value = mock_model
    mock_get_peft.return_value = mock_model

    from src.training.model_trainer import ModelTrainer

    trainer = ModelTrainer(config=training_config, model_config=model_config)
    trainer.setup_peft_model()

    mock_get_peft.assert_called_once()
    lora_cfg = mock_get_peft.call_args[0][1]
    assert lora_cfg.r == training_config.lora_r
    assert lora_cfg.lora_alpha == training_config.lora_alpha
    assert lora_cfg.lora_dropout == training_config.lora_dropout
    assert set(lora_cfg.target_modules) == set(training_config.target_modules)


@patch("src.training.model_trainer.SFTTrainer")
@patch("src.training.model_trainer.load_from_disk")
@patch("src.training.model_trainer.AutoModelForCausalLM.from_pretrained")
@patch("src.training.model_trainer.os.makedirs")
def test_train_calls_sfttrainer(
    mock_makedirs,
    mock_from_pretrained,
    mock_load_disk,
    mock_sft_cls,
    training_config,
    model_config,
):
    """train() loads the dataset and calls SFTTrainer.train()."""
    mock_from_pretrained.return_value = MagicMock()
    mock_dataset = {"train": list(range(100)), "validation": list(range(10))}
    mock_load_disk.return_value = mock_dataset

    mock_trainer_instance = MagicMock()
    mock_sft_cls.return_value = mock_trainer_instance

    from src.training.model_trainer import ModelTrainer

    obj = ModelTrainer(config=training_config, model_config=model_config)
    returned_trainer = obj.train()

    mock_load_disk.assert_called_once_with(str(training_config.dataset_path))
    mock_trainer_instance.train.assert_called_once()
    assert returned_trainer == mock_trainer_instance


@patch("src.training.model_trainer.AutoModelForCausalLM.from_pretrained")
@patch("src.training.model_trainer.os.makedirs")
def test_save_calls_trainer_save_model(
    mock_makedirs, mock_from_pretrained, training_config, model_config
):
    """save() calls trainer.save_model() with the correct path."""
    mock_from_pretrained.return_value = MagicMock()

    from src.training.model_trainer import ModelTrainer

    obj = ModelTrainer(config=training_config, model_config=model_config)
    mock_trainer = MagicMock()
    result_path = obj.save(mock_trainer)

    mock_trainer.save_model.assert_called_once_with(str(training_config.root_dir))
    assert result_path == Path(training_config.root_dir)


@patch("src.training.model_trainer.SFTTrainer")
@patch("src.training.model_trainer.load_from_disk")
@patch("src.training.model_trainer.AutoModelForCausalLM.from_pretrained")
@patch("src.training.model_trainer.os.makedirs")
def test_train_oom_raises_model_error(
    mock_makedirs,
    mock_from_pretrained,
    mock_load_disk,
    mock_sft_cls,
    training_config,
    model_config,
):
    """train() converts CUDA OOM to ModelError with a friendly message."""
    mock_from_pretrained.return_value = MagicMock()
    mock_load_disk.return_value = {"train": [1], "validation": [1]}

    mock_trainer_instance = MagicMock()
    mock_trainer_instance.train.side_effect = torch.cuda.OutOfMemoryError("CUDA OOM")
    mock_sft_cls.return_value = mock_trainer_instance

    from src.training.model_trainer import ModelTrainer
    from src.utils.exception import ModelError

    obj = ModelTrainer(config=training_config, model_config=model_config)
    with pytest.raises(ModelError, match="OOM"):
        obj.train()
