import os
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from src.data.data_validation import DataValidation
from src.entity.config_entity import DataTransformationConfig, DataValidationConfig
from src.utils.config import ConfigurationManager
from src.utils.exception import he_raise
from src.utils.logger import logger


SYSTEM_PROMPT = (
    "You are a Python bug detection expert. "
    "When given a piece of Python code, you must identify the bug type, "
    "provide a high-level description of the issue, and supply the corrected code."
)


class DataTransformation:
    """
    Handles data transformation for fine-tuning causal LLMs.
    Includes: prompt formatting, tokenization, and train/validation split.

    Uses `tokenizer.apply_chat_template` — compatible with any model
    (Qwen2.5, Llama 3, Phi-3, etc.) without hardcoded special tokens.
    """

    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Tokenizer loaded: {config.tokenizer_name}")

    def format_prompt(self, row: dict) -> str:
        """
        Formats a single data row into the model's native chat template.

        Uses `tokenizer.apply_chat_template` so the output is correct
        for any model (Qwen2.5, Llama 3, etc.) without hardcoded tokens.

        Args:
            row: Dictionary with keys 'Faulty Code', 'Bug_Type',
                 'High-Level Description', 'Fault Free Code'.
        Returns:
            Formatted prompt string ready for tokenization.
        """
        user_message = (
            "Analyze the following Python code and identify the bug.\n\n"
            f"```python\n{row['Faulty Code']}\n```"
        )
        assistant_message = (
            f"**Bug Type**: {row['Bug_Type']}\n\n"
            f"**Description**: {row['High-Level Description']}\n\n"
            f"**Fixed Code**:\n```python\n{row['Fault Free Code']}\n```"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ]

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    @he_raise
    def load_data(self, validation_config: DataValidationConfig) -> pd.DataFrame:
        """
        Loads data from the ingestion output, only if data validation passes.

        Args:
            validation_config: DataValidationConfig to run validation first.
        Returns:
            pd.DataFrame: The loaded data.
        """
        # Run data validation first
        data_validation = DataValidation(config=validation_config)
        is_valid = data_validation.validate_all_files_exist()

        if not is_valid:
            raise ValueError(
                "Data validation failed. Cannot proceed with data transformation."
            )

        logger.info(f"Loading data from: {self.config.source_data_file}")
        df = pd.read_csv(self.config.source_data_file, sep="\t")
        logger.info(f"Loaded {len(df)} samples.")

        # Drop rows with missing essential columns
        required_cols = ["Faulty Code", "Fault Free Code", "Bug_Type", "High-Level Description"]
        df = df.dropna(subset=required_cols)
        logger.info(f"After dropping NaN rows: {len(df)} samples remaining.")

        return df

    @he_raise
    def transform(self, df: pd.DataFrame) -> DatasetDict:
        """
        Formats prompts, tokenizes, and splits the dataset.

        Args:
            df: DataFrame with raw data.
        Returns:
            DatasetDict with 'train' and 'validation' splits.
        """
        logger.info("Formatting prompts...")
        df["text"] = df.apply(self.format_prompt, axis=1)

        logger.info("Converting to HuggingFace Dataset...")
        dataset = Dataset.from_pandas(df[["text"]], preserve_index=False)

        # Tokenize
        logger.info(f"Tokenizing with max_length={self.config.max_length}...")

        def tokenize_fn(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.max_length,
                padding="max_length",
            )

        tokenized_dataset = dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing",
        )

        # Train / Validation split
        logger.info(f"Splitting dataset (test_size={self.config.test_size})...")
        split = tokenized_dataset.train_test_split(
            test_size=self.config.test_size, seed=42
        )
        dataset_dict = DatasetDict(
            {"train": split["train"], "validation": split["test"]}
        )

        logger.info(
            f"Split complete — Train: {len(dataset_dict['train'])}, "
            f"Validation: {len(dataset_dict['validation'])}"
        )
        return dataset_dict

    @he_raise
    def save(self, dataset_dict: DatasetDict) -> Path:
        """
        Saves the processed dataset to disk.

        Args:
            dataset_dict: DatasetDict with 'train' and 'validation' splits.
        Returns:
            Path: The path where the dataset was saved.
        """
        os.makedirs(self.config.root_dir, exist_ok=True)
        save_path = self.config.root_dir

        dataset_dict.save_to_disk(str(save_path))
        logger.info(f"Dataset saved to: {save_path}")
        return save_path


if __name__ == "__main__":
    try:
        config_manager = ConfigurationManager()

        validation_config = config_manager.get_data_validation_config()
        transformation_config = config_manager.get_data_transformation_config()

        transformer = DataTransformation(config=transformation_config)

        # Step 1: Load data (with validation check)
        df = transformer.load_data(validation_config=validation_config)

        # Step 2: Transform (format, tokenize, split)
        dataset_dict = transformer.transform(df)

        # Step 3: Save to disk
        output_path = transformer.save(dataset_dict)
        logger.info(f"Data transformation complete. Output: {output_path}")

    except Exception:
        # Error is already logged by @he_raise
        pass
