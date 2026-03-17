# AGENTS.md

This file provides guidelines for AI agents working in this repository.

## Project Overview

This is a fine-tuning project for bug detection using Llama 3.1 8B with QLoRA (4-bit quantization) + SFTTrainer. The pipeline includes data ingestion, validation, transformation, training, and evaluation.

## Build/Lint/Test Commands

### Testing
```bash
# Run all tests
pytest

# Run a single test file
pytest tests/training/test_model_trainer.py

# Run a single test by name
pytest tests/training/test_model_trainer.py::test_init_creates_directories_and_loads_model

# Run tests with verbose output
pytest -v

# Run tests matching a pattern
pytest -k "test_train"
```

### Linting/Type Checking
```bash
# Run ruff linter (if installed)
ruff check .

# Run mypy type checker (if installed)
mypy src/
```

### Development
```bash
# Install package in editable mode
pip install -e .

# Run the training pipeline
python train.py
```

## Code Style Guidelines

### Imports
- Use absolute imports from the `src` package (e.g., `from src.training.model_trainer import ModelTrainer`)
- Group imports: standard library → third-party → local application
- Use `from typing import Optional` for type hints

### Formatting
- Maximum line length: 100 characters
- Use 4 spaces for indentation
- Use blank lines to separate logical sections within functions

### Types
- Use Python type hints for all function parameters and return types
- Use `Path` from `pathlib` for file paths
- Use `dataclass` with `frozen=True` for configuration entities
- Example:
  ```python
  from pathlib import Path
  from typing import Optional
  
  def load_model(self) -> AutoModelForCausalLM:
      ...
  ```

### Naming Conventions
- **Classes**: PascalCase (e.g., `ModelTrainer`, `LLMProvider`)
- **Functions/methods**: snake_case (e.g., `load_model`, `setup_peft_model`)
- **Constants**: UPPER_SNAKE_CASE
- **Private methods**: prefix with underscore (e.g., `_internal_method`)
- **Config classes**: suffix with `Config` (e.g., `ModelConfig`, `TrainingConfig`)

### Error Handling
- Use custom exception hierarchy inheriting from `AppError` (see `src/utils/exception.py`)
- Use the `@he_raise` decorator for functions that should log and re-raise exceptions
- Catch specific exceptions (e.g., `torch.cuda.OutOfMemoryError`) and convert to custom exceptions with user-friendly messages
- Example:
  ```python
  from src.utils.exception import ModelError, he_raise
  
  @he_raise
  def train(self):
      try:
          ...
      except torch.cuda.OutOfMemoryError as e:
          logger.error("CUDA OOM error message")
          raise ModelError("Friendly message") from e
  ```

### Logging
- Use the logger from `src.utils.logger`
- Log at appropriate levels: `logger.debug()`, `logger.info()`, `logger.warning()`, `logger.error()`, `logger.critical()`
- Include relevant context in log messages

### Testing
- Use `pytest` with fixtures
- Use `unittest.mock` for mocking external dependencies
- Follow AAA pattern: Arrange → Act → Assert
- Mock at the correct level (e.g., patch `src.training.model_trainer.AutoModelForCausalLM.from_pretrained` when testing model loading)
- Example:
  ```python
  @patch("src.training.model_trainer.AutoModelForCausalLM.from_pretrained")
  @patch("src.training.model_trainer.os.makedirs")
  def test_init_creates_directories_and_loads_model(
      mock_makedirs, mock_from_pretrained, training_config, model_config
  ):
      mock_model = MagicMock()
      mock_from_pretrained.return_value = mock_model
      
      trainer = ModelTrainer(config=training_config, model_config=model_config)
      
      assert trainer.model == mock_model
  ```

### Project Structure
```
src/
├── constants/       # Constants
├── data/            # Data loading/processing
├── entity/          # Configuration dataclasses
├── evaluation/      # Model evaluation
├── models/          # Model utilities (deprecated - merged into training)
├── training/        # Training logic (model_trainer.py)
└── utils/           # Utilities (config, exceptions, logger)

tests/
├── models/          # Model tests (deprecated - merged into training)
└── training/        # Training tests
```

### Configuration
- Use YAML files in `configs/` for configuration
- Use dataclasses in `src/entity/config_entity.py` for type-safe config access
- Configuration should be immutable (`frozen=True` dataclass)

### Documentation
- Use Google-style docstrings for classes and public methods
- Include docstrings for all public classes and functions
- Example:
  ```python
  def load_model(self) -> AutoModelForCausalLM:
      """
      Loads the model from Hugging Face with optimal settings.
      
      Returns:
          AutoModelForCausalLM: The loaded model instance.
      """
  ```

### General Principles
- Keep functions focused and single-purpose
- Use dependency injection for testability
- Avoid hardcoding; use configuration
- Clean up resources (use context managers where appropriate)
