from typing import Optional

import torch
from transformers import BitsAndBytesConfig

from src.utils.logger import logger


def build_quantization_config(use_4bit: bool) -> Optional[BitsAndBytesConfig]:
    """
    Builds the 4-bit quantization (QLoRA) configuration.

    Args:
        use_4bit: Whether to enable 4-bit quantization.

    Returns:
        BitsAndBytesConfig if use_4bit is True, otherwise None.
    """
    if not use_4bit:
        logger.debug("4-bit quantization disabled. Using full precision.")
        return None

    logger.info("Building 4-bit QLoRA quantization config (nf4, bfloat16 compute).")
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
