from pathlib import Path

import yaml

from src.constants import CONFIG_FILE_PATH
from src.entity.config_entity import AppConfig, DataIngestionConfig


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
