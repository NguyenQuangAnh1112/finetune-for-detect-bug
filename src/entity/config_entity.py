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
class AppConfig:
    data_ingestion: DataIngestionConfig
    data_validation: DataValidationConfig
    data_transformation: DataTransformationConfig
