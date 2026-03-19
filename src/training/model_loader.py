import os

import torch
from transformers import AutoModelForCausalLM

from src.entity.config_entity import ModelConfig
from src.training.quantization_config import build_quantization_config
from src.utils.exception import ModelError
from src.utils.logger import logger


def load_model(model_config: ModelConfig) -> AutoModelForCausalLM:
    """
    Loads the base LLM from HuggingFace with optional QLoRA quantization.

    Args:
        model_config: Configuration for the model (name, cache dir, quantization flag).

    Returns:
        The loaded AutoModelForCausalLM instance.

    Raises:
        ModelError: If the model fails to load for any reason.
    """
    try:
        os.makedirs(model_config.cache_dir, exist_ok=True)
        os.makedirs(model_config.root_dir, exist_ok=True)
        logger.debug(
            f"Ensured directories exist: cache='{model_config.cache_dir}', "
            f"root='{model_config.root_dir}'"
        )
    except OSError as e:
        logger.error(f"Cannot create model directories: {e}")
        raise ModelError("Failed to create model cache/root directories.") from e

    quantization_config = build_quantization_config(model_config.use_4bit)

    logger.info(f"Loading model '{model_config.model_name}' from HuggingFace...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            cache_dir=model_config.cache_dir,
            torch_dtype=torch.bfloat16,
        )
    except OSError as e:
        logger.error(
            f"Model '{model_config.model_name}' not found or cannot be accessed. "
            f"Check model name and network. Error: {e}"
        )
        raise ModelError(f"Model '{model_config.model_name}' could not be loaded.") from e
    except Exception as e:
        logger.error(f"Unexpected error while loading model: {e}")
        raise ModelError("Unexpected failure during model loading.") from e

    logger.info(
        f"Model loaded successfully. Cached at: '{model_config.cache_dir}'"
    )
    return model  # type: ignore[return-value]
