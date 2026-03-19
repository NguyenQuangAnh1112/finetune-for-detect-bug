from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from src.data.data_transformation import DataTransformation
from src.entity.config_entity import ModelConfig, TrainingConfig, DataTransformationConfig
from src.training.trainer_runner import run_training


@contextmanager
def _dummy_run():
    yield


def _build_mock_tokenizer() -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.pad_token = None
    tokenizer.eos_token = "<eos>"
    tokenizer.apply_chat_template.return_value = "prompt"

    def _tokenize(texts, truncation, max_length, padding):
        batch_size = len(texts)
        return {
            "input_ids": [[1, 2]] * batch_size,
            "attention_mask": [[1, 1]] * batch_size,
        }

    tokenizer.side_effect = _tokenize
    return tokenizer


def test_training_pipeline_integration(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "Faulty Code": ["print(1)", "print(2)"],
            "Fault Free Code": ["print(1)", "print(2)"],
            "Bug_Type": ["none", "none"],
            "High-Level Description": ["ok", "ok"],
        }
    )

    transform_config = DataTransformationConfig(
        root_dir=tmp_path / "dataset",
        source_data_file=tmp_path / "data.tsv",
        tokenizer_name="mock-tokenizer",
        max_length=32,
        test_size=0.5,
    )

    with patch(
        "src.data.data_transformation.AutoTokenizer.from_pretrained",
        return_value=_build_mock_tokenizer(),
    ):
        transformer = DataTransformation(config=transform_config)
        dataset_dict = transformer.transform(df)
        transformer.save(dataset_dict)

    training_config = TrainingConfig(
        root_dir=tmp_path / "training",
        dataset_path=transform_config.root_dir,
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
    model_config = ModelConfig(
        root_dir=tmp_path / "model",
        model_name="mock-model/Mock-LLM-8B",
        cache_dir=tmp_path / "cache",
        use_4bit=True,
    )

    mock_trainer = MagicMock()
    with patch("src.training.trainer_runner.SFTTrainer", return_value=mock_trainer), patch(
        "src.training.trainer_runner.mlflow"
    ) as mock_mlflow:
        mock_mlflow.start_run.return_value = _dummy_run()
        trainer = run_training(MagicMock(), training_config, model_config)

    mock_trainer.train.assert_called_once()
    assert trainer == mock_trainer
