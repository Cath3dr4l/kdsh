
# Project Documentation

## Overview
This project consists of three main components:
1. **Embedder**: Responsible for embedding tasks.
2. **Indexer (Vectorstore)**: Handles the indexing of embeddings for efficient storage and retrieval using the Pathway Drive connector and Document Store.
3. **Classifier**: Provides classification services by leveraging prebuilt classifiers.

## Setup and Execution

### Running the Embedder
1. Navigate to the `embedder` directory:
   ```bash
   cd embedder
   ```
2. Run the embedder service:
   ```bash
   modal run modal_service.py
   ```

### Running the Indexer (Vectorstore)
1. Navigate to the `indexer` directory:
   ```bash
   cd indexer
   ```
2. Install the required dependencies using Poetry:
   ```bash
   poetry install
   ```
3. Run the indexer service:
   ```bash
   poetry run python services/indexer.py
   ```

### Running the Classifier
1. Import any of the classifiers from the `services` directory in your Python code.
2. Use the `classify` method to perform classification tasks. Example:
   ```python
   from services.some_classifier import SomeClassifier

   classifier = SomeClassifier()
   result = await classifier.classify(input_content)
   print(result)
   ```

## Project Structure
- **`embedder/`**: Contains scripts and services for embedding tasks.
- **`indexer/`**: Contains the vectorstore implementation for efficient storage and retrieval of embeddings.
- **`services/`**: Includes various classification models and utilities.

## Prerequisites
Ensure the following are installed:
- Python (compatible version as per `pyproject.toml` in the `indexer` directory)
- Poetry (for dependency management in the `indexer` module)
- Modal (for running the embedder service)

## Dependencies
 Managed using Poetry. Install them with:
  ```bash
  poetry install
  ```
-
## Usage
- **Embedder**: Generates embeddings from input data.
- **Indexer**: Stores embeddings for quick and efficient retrieval. uses pathway drive connector and document store.
- **Classifier**: Performs classification tasks using predefined classifiers.

