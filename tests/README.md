# Tests

This directory contains unit and integration tests for the credit underwriting system.

## Running Tests

Once tests are added, they can be run with Python's built-in unittest module:

```bash
python -m pytest tests/  # If pytest is installed
# or
python -m unittest discover tests/  # Using built-in unittest
```

## Test Structure

Suggested test organization:
- `test_data/`: Tests for data generation and processing
- `test_models/`: Tests for model training and prediction
- `test_integration/`: End-to-end pipeline tests
