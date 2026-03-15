import os
from pathlib import Path

from datasets import load_dataset

from src.entity.config_entity import DataIngestionConfig
from src.utils.config import ConfigurationManager
from src.utils.exception import he_raise
from src.utils.logger import logger


class DataIngestion:
    """
    Handles retrieval and storage of data from external sources.
    """

    def __init__(self, config: DataIngestionConfig):
        self.config = config

    @he_raise
    def download_data(self) -> Path:
        """
        Downloads data from HuggingFace Hub and saves it locally.

        Returns:
            Path: The path to the downloaded data file.
        """
        logger.info(f"Starting data ingestion from: {self.config.huggingface_source_url}")

        # Create root directory if it doesn't exist
        os.makedirs(self.config.root_dir, exist_ok=True)
        logger.debug(f"Ensured directory exists: {self.config.root_dir}")

        # Load dataset from HuggingFace
        dataset = load_dataset(self.config.huggingface_source_url)
        logger.info(f"Dataset downloaded successfully from HuggingFace.")

        # Save to local file (assuming TSV as per config)
        # Note: The 'train' split is usually what we need, but this can be adjusted
        df = dataset["train"].to_pandas()
        df.to_csv(self.config.local_data_file, sep="\t", index=False)

        logger.info(f"Data saved locally at: {self.config.local_data_file}")
        return self.config.local_data_file


if __name__ == "__main__":
    try:
        config_manager = ConfigurationManager()
        ingestion_config = config_manager.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=ingestion_config)
        data_ingestion.download_data()
    except Exception:
        # Error is already logged by @he_raise in the method
        pass
