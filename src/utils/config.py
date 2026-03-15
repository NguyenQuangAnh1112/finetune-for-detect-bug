from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

from src.constants import CONFIG_FILE_PATH
from src.entity.config_entity import (
    AppConfig,
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
)


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
