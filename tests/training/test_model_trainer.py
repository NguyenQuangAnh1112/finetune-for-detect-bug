from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.entity.config_entity import ModelConfig, TrainingConfig


@pytest.fixture
def training_config() -> TrainingConfig:
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
def model_config() -> ModelConfig:
    return ModelConfig(
        root_dir=Path("/mock/artifacts/model"),
        model_name="mock-model/Mock-LLM-8B",
        cache_dir=Path("/mock/cache/models"),
        use_4bit=True,
    )


@patch("src.training.model_trainer.os.makedirs")
@patch("src.training.model_trainer.load_model")
def test_init_creates_training_dir_and_loads_model(
    mock_load_model, mock_makedirs, training_config, model_config
) -> None:
    mock_model = MagicMock()
    mock_load_model.return_value = mock_model

    from src.training.model_trainer import ModelTrainer

    trainer = ModelTrainer(config=training_config, model_config=model_config)

    mock_makedirs.assert_called_once_with(training_config.root_dir, exist_ok=True)
    mock_load_model.assert_called_once_with(model_config)
    assert trainer.model == mock_model


@patch("src.training.model_trainer.apply_lora")
@patch("src.training.model_trainer.load_model")
@patch("src.training.model_trainer.os.makedirs")
def test_setup_peft_model_applies_lora(
    mock_makedirs, mock_load_model, mock_apply_lora, training_config, model_config
) -> None:
    base_model = MagicMock()
    peft_model = MagicMock()
    mock_load_model.return_value = base_model
    mock_apply_lora.return_value = peft_model

    from src.training.model_trainer import ModelTrainer

    trainer = ModelTrainer(config=training_config, model_config=model_config)
    trainer.setup_peft_model()

    mock_apply_lora.assert_called_once_with(base_model, training_config)
    assert trainer.model == peft_model


@patch("src.training.model_trainer.run_training")
@patch("src.training.model_trainer.load_model")
@patch("src.training.model_trainer.os.makedirs")
def test_train_delegates_to_trainer_runner(
    mock_makedirs, mock_load_model, mock_run_training, training_config, model_config
) -> None:
    mock_model = MagicMock()
    mock_trainer = MagicMock()
    mock_load_model.return_value = mock_model
    mock_run_training.return_value = mock_trainer

    from src.training.model_trainer import ModelTrainer

    trainer_obj = ModelTrainer(config=training_config, model_config=model_config)
    returned_trainer = trainer_obj.train()

    mock_run_training.assert_called_once_with(mock_model, training_config, model_config)
    assert returned_trainer == mock_trainer


@patch("src.training.model_trainer.load_model")
@patch("src.training.model_trainer.os.makedirs")
def test_save_calls_trainer_save_model(
    mock_makedirs, mock_load_model, training_config, model_config
) -> None:
    mock_load_model.return_value = MagicMock()

    from src.training.model_trainer import ModelTrainer

    trainer_obj = ModelTrainer(config=training_config, model_config=model_config)
    mock_trainer = MagicMock()
    result_path = trainer_obj.save(mock_trainer)

    mock_trainer.save_model.assert_called_once_with(str(training_config.root_dir))
    assert result_path == Path(training_config.root_dir)
