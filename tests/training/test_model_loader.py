from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from transformers import BitsAndBytesConfig

from src.entity.config_entity import ModelConfig
from src.training.model_loader import load_model
from src.training.quantization_config import build_quantization_config


def test_build_quantization_config_enabled() -> None:
    config = build_quantization_config(use_4bit=True)
    assert isinstance(config, BitsAndBytesConfig)
    assert config.load_in_4bit is True
    assert config.bnb_4bit_quant_type == "nf4"
    assert config.bnb_4bit_compute_dtype == torch.bfloat16
    assert config.bnb_4bit_use_double_quant is True


def test_build_quantization_config_disabled() -> None:
    config = build_quantization_config(use_4bit=False)
    assert config is None


def test_load_model_with_4bit(tmp_path: Path) -> None:
    model_config = ModelConfig(
        root_dir=tmp_path / "model",
        model_name="mock-model/Mock-LLM-8B",
        cache_dir=tmp_path / "cache",
        use_4bit=True,
    )

    mock_model = MagicMock()
    with patch(
        "src.training.model_loader.AutoModelForCausalLM.from_pretrained",
        return_value=mock_model,
    ) as mock_from_pretrained:
        model = load_model(model_config)

    assert model == mock_model
    call_args, call_kwargs = mock_from_pretrained.call_args
    assert call_args[0] == model_config.model_name
    assert call_kwargs["device_map"] == "auto"
    assert call_kwargs["cache_dir"] == model_config.cache_dir
    assert call_kwargs["torch_dtype"] == torch.bfloat16
    assert isinstance(call_kwargs["quantization_config"], BitsAndBytesConfig)


def test_load_model_without_4bit(tmp_path: Path) -> None:
    model_config = ModelConfig(
        root_dir=tmp_path / "model",
        model_name="mock-model/Mock-LLM-8B",
        cache_dir=tmp_path / "cache",
        use_4bit=False,
    )

    mock_model = MagicMock()
    with patch(
        "src.training.model_loader.AutoModelForCausalLM.from_pretrained",
        return_value=mock_model,
    ) as mock_from_pretrained:
        model = load_model(model_config)

    assert model == mock_model
    call_args, call_kwargs = mock_from_pretrained.call_args
    assert call_args[0] == model_config.model_name
    assert call_kwargs["device_map"] == "auto"
    assert call_kwargs["cache_dir"] == model_config.cache_dir
    assert call_kwargs["torch_dtype"] == torch.bfloat16
    assert call_kwargs["quantization_config"] is None
