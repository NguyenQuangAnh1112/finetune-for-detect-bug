from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import PreTrainedModel

from src.entity.config_entity import TrainingConfig
from src.utils.exception import ModelError
from src.utils.logger import logger


def apply_lora(model: PreTrainedModel, config: TrainingConfig) -> PeftModel:
    """
    Wraps the base model with a LoRA adapter using the provided training config.

    Args:
        model: The base pre-trained model to adapt.
        config: TrainingConfig containing LoRA hyperparameters.

    Returns:
        The model wrapped with a PEFT LoRA adapter.

    Raises:
        ModelError: If target_modules are invalid or PEFT setup fails.
    """
    logger.info(
        f"Applying LoRA adapter — r={config.lora_r}, alpha={config.lora_alpha}, "
        f"dropout={config.lora_dropout}, target_modules={config.target_modules}"
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
    )

    try:
        peft_model = get_peft_model(model, lora_config)
    except ValueError as e:
        logger.error(
            f"PEFT setup failed — invalid target_modules: {config.target_modules}. "
            f"Ensure they exist in the model architecture. Error: {e}"
        )
        raise ModelError("LoRA application failed due to invalid target_modules.") from e
    except Exception as e:
        logger.error(f"Unexpected error during PEFT setup: {e}")
        raise ModelError("Unexpected failure during LoRA adapter setup.") from e

    trainable, total = peft_model.get_nb_trainable_parameters()
    logger.info(
        f"LoRA applied successfully. Trainable params: {trainable:,} / "
        f"Total: {total:,} ({100 * trainable / total:.4f}%)"
    )
    return peft_model
