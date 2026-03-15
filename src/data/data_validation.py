import os

from src.entity.config_entity import DataValidationConfig
from src.utils.config import ConfigurationManager
from src.utils.exception import he_raise
from src.utils.logger import logger


class DataValidation:
    """
    Validates the existence and integrity of ingested data.
    """

    def __init__(self, config: DataValidationConfig):
        self.config = config

    @he_raise
    def validate_all_files_exist(self) -> bool:
        """
        Validates that all required files exist on disk.

        Returns:
            bool: True if all files exist, False otherwise.
        """
        validation_status = None
        required_files = self.config.all_required_files

        logger.info(f"Starting data validation for {len(required_files)} files.")

        try:
            # Create root directory for validation artifacts
            os.makedirs(self.config.root_dir, exist_ok=True)

            for file_path in required_files:
                if not os.path.exists(file_path):
                    validation_status = False
                    logger.error(f"Validation failed: File missing -> {file_path}")
                else:
                    validation_status = True if validation_status is not False else False
                    logger.debug(f"Validation passed: File exists -> {file_path}")

            # If no files missing, it's True
            if validation_status is None:
                validation_status = False  # No files to check?

            # Write status to file
            with open(self.config.status_file, "w") as f:
                f.write(f"Validation status: {validation_status}")

            if validation_status:
                logger.info("Data validation completed successfully.")
            else:
                logger.warning("Data validation failed. Some files are missing.")

            return validation_status

        except Exception as e:
            logger.exception(f"Unexpected error during data validation: {e}")
            raise e


if __name__ == "__main__":
    try:
        config_manager = ConfigurationManager()
        validation_config = config_manager.get_data_validation_config()
        data_validation = DataValidation(config=validation_config)
        data_validation.validate_all_files_exist()
    except Exception:
        # Error is already logged by @he_raise or logger.exception
        pass
