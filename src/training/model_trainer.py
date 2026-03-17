import os
from pathlib import Path

from datasets import load_from_disk
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, PreTrainedModel, TrainingArguments
from trl import SFTTrainer

from src.entity.config_entity import ModelConfig, TrainingConfig
from src.training.model_loader import load_model
from src.training.peft_setup import apply_lora
from src.training.trainer_runner import run_training
from src.utils.config import ConfigurationManager
from src.utils.exception import ModelError, he_raise
from src.utils.logger import logger


class ModelTrainer:
    """
    Orchestrates the QLoRA fine-tuning pipeline for Llama 3.1 8B.

    Responsibilities:
        - Delegates model loading to `model_loader.load_model`
        - Delegates LoRA setup to `peft_setup.apply_lora`
        - Delegates training to `trainer_runner.run_training`
        - Handles adapter persistence via `save`

    Input : artifacts/data_transformation/ (tokenized DatasetDict)
    Output: artifacts/training/ (LoRA adapter weights)
    """

    def __init__(self, config: TrainingConfig, model_config: ModelConfig) -> None:
        logger.info("Initializing ModelTrainer...")
        os.makedirs(config.root_dir, exist_ok=True)
        self.config = config
        self.model_config = model_config

        try:
            self.model = load_model(model_config)
        except ModelError:
            raise
        except Exception as e:
            logger.error(f"Unexpected failure during ModelTrainer init: {e}")
            raise ModelError("Model loading failed during ModelTrainer init.") from e

        logger.info("ModelTrainer is ready.")

    @he_raise
    def setup_peft_model(self) -> None:
        """Wraps the loaded model with a LoRA adapter (delegates to peft_setup)."""
        self.model = apply_lora(self.model, self.config)

    @he_raise
    def train(self) -> SFTTrainer:
        """Loads dataset and runs SFT training (delegates to trainer_runner)."""
        return run_training(self.model, self.config, self.model_config)

    @he_raise
    def save(self, trainer: SFTTrainer) -> Path:
        """
        Saves the LoRA adapter weights to artifacts/training/.

        Args:
            trainer: The SFTTrainer instance returned by `train()`.

        Returns:
            The path where the adapter was saved.

        Raises:
            OSError: If the disk write fails.
        """
        save_path = Path(self.config.root_dir)
        logger.info(f"Saving LoRA adapter to: '{save_path}'")
        try:
            trainer.save_model(str(save_path))
        except OSError as e:
            logger.error(f"Failed to save adapter — disk write error: {e}")
            raise

        logger.info(f"Adapter saved successfully at: '{save_path}'")
        return save_path


if __name__ == "__main__":
    try:
        config_manager = ConfigurationManager()
        model_config = config_manager.get_model_config()
        training_config = config_manager.get_training_config()

        trainer_obj = ModelTrainer(config=training_config, model_config=model_config)
        trainer_obj.setup_peft_model()
        trainer = trainer_obj.train()
        trainer_obj.save(trainer)

    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise
