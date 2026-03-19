"""CLI entry point for the training pipeline."""

import argparse
from pathlib import Path

from src.training.model_trainer import ModelTrainer
from src.utils.config import ConfigurationManager
from src.utils.exception import ModelError
from src.utils.logger import logger


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the QLoRA training pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config.yaml"),
        help="Path to YAML config file",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    logger.info(f"Using config: {args.config}")
    config_manager = ConfigurationManager(args.config)

    try:
        model_config = config_manager.get_model_config()
        training_config = config_manager.get_training_config()

        trainer_obj = ModelTrainer(config=training_config, model_config=model_config)
        trainer_obj.setup_peft_model()
        trainer = trainer_obj.train()
        trainer_obj.save(trainer)
    except ModelError as e:
        logger.error(f"Training failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in training pipeline: {e}")
        raise


if __name__ == "__main__":
    main()
