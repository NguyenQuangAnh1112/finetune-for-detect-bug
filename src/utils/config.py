from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.constants import CONFIG_FILE_PATH
from src.entity.config_entity import (
    AppConfig,
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    ModelConfig,
    TrainingConfig,
)

load_dotenv()


class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH):
        with open(config_filepath) as f:
            self.config = yaml.safe_load(f)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.get("data_ingestion", {})

        return DataIngestionConfig(
            root_dir=Path(config.get("root_dir")),
            huggingface_source_url=config.get("huggingface_source_url"),
            local_data_file=Path(config.get("local_data_file")),
        )

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.get("data_validation", {})

        return DataValidationConfig(
            root_dir=Path(config.get("root_dir")),
            status_file=Path(config.get("status_file")),
            all_required_files=config.get("all_required_files", []),
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.get("data_transformation", {})

        return DataTransformationConfig(
            root_dir=Path(config.get("root_dir")),
            source_data_file=Path(config.get("source_data_file")),
            tokenizer_name=config.get("tokenizer_name"),
            max_length=config.get("max_length"),
            test_size=config.get("test_size"),
        )

    def get_model_config(self) -> ModelConfig:
        config = self.config.get("model", {})

        return ModelConfig(
            root_dir=Path(config.get("root_dir")),
            model_name=config.get("model_name"),
            cache_dir=Path(config.get("cache_dir")),
            use_4bit=config.get("use_4bit", True),
        )

    def get_training_config(self) -> TrainingConfig:
        config = self.config.get("training", {})

        return TrainingConfig(
            root_dir=Path(config.get("root_dir")),
            dataset_path=Path(config.get("dataset_path")),
            num_train_epochs=config.get("num_train_epochs", 3),
            per_device_train_batch_size=config.get("per_device_train_batch_size", 2),
            per_device_eval_batch_size=config.get("per_device_eval_batch_size", 2),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
            learning_rate=config.get("learning_rate", 2e-4),
            logging_steps=config.get("logging_steps", 25),
            save_steps=config.get("save_steps", 100),
            eval_steps=config.get("eval_steps", 100),
            warmup_ratio=config.get("warmup_ratio", 0.05),
            lora_r=config.get("lora_r", 16),
            lora_alpha=config.get("lora_alpha", 32),
            lora_dropout=config.get("lora_dropout", 0.05),
            target_modules=config.get("target_modules", ["q_proj", "v_proj"]),
        )
