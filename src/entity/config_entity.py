from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    huggingface_source_url: str
    local_data_file: Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    status_file: Path
    all_required_files: list


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    source_data_file: Path
    tokenizer_name: str
    max_length: int
    test_size: float


@dataclass(frozen=True)
class ModelConfig:
    root_dir: Path
    model_name: str
    cache_dir: Path
    use_4bit: bool


@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    dataset_path: Path
    num_train_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    logging_steps: int
    save_steps: int
    eval_steps: int
    warmup_ratio: float
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list


@dataclass(frozen=True)
class AppConfig:
    data_ingestion: DataIngestionConfig
    data_validation: DataValidationConfig
    data_transformation: DataTransformationConfig
    model: ModelConfig
    training: TrainingConfig
