# ARC2 Indexer Service

This directory contains the data ingestion and indexing pipeline for the **ARC2 (Agentic AI Research Review and Conference Classification)** project. It uses the [Pathway](https://pathway.com/) framework to create and maintain a real-time hybrid search index for the RAG-based classifier agent.

## 🚀 Overview

The Indexer Service is a critical component of the **SCRIBE** architecture. It is responsible for:

1.  **Data Ingestion**: Automatically reading research papers and conference data from a specified source (e.g., Google Drive).
2.  **Data Processing**: Parsing documents, splitting them into manageable chunks, and generating embeddings using the [Embedder Service](../embedder/README.md).
3.  **Indexing**: Building a hybrid search index that combines lexical search (BM25) and semantic search (USearchKNN) for robust and accurate retrieval.

This service runs as a background process, ensuring that the RAG agent always has access to the most up-to-date information.

## 🛠️ Tech Stack

-   Python 3.11
-   [Pathway](https://pathway.com/)
-   [LangChain](https://www.langchain.com/)

## ⚙️ Setup and Execution

### Prerequisites

-   Python 3.11+
-   Poetry
-   Access to the [Embedder Service](../embedder/README.md).
-   A Google Drive folder containing the source documents.

### Installation

1.  Navigate to the `indexer` directory:
    ```bash
    cd indexer
    ```
2.  Install dependencies using Poetry:
    ```bash
    poetry install
    ```
3.  Set up authentication for Google Drive by following the Pathway documentation. You will need to create a `google_drive_creds.json` file in this directory.

### Running the Service

To run the indexer service, use the following command from the `indexer` directory:

```bash
poetry run python services/indexer.py
```

The service will start, connect to Google Drive, and begin processing the documents into a searchable index. The index will be persisted to the `storage/` directory.

## 🏛️ Architecture Deep Dive

The indexer service is built entirely on the Pathway framework, which allows it to process data in a streaming fashion.

- **[`Indexer`](services/indexer.py:13)**: This is the main class that orchestrates the entire indexing pipeline. It initializes the data connectors and the document store, and then runs the Pathway graph.

- **Data Ingestion**:
    - **[`DriveConnector`](services/drive_connector.py)**: This class (not shown, but part of the project) is a custom Pathway connector that reads documents from a specified Google Drive folder. It produces a `pw.Table` of documents.

- **Indexing and Retrieval (`DocumentStoreServerWrapper`)**:
    - The [`DocumentStoreServerWrapper`](services/document_store.py:38) class in `services/document_store.py` is the core of the indexing logic. It sets up and configures the `DocumentStore`.
    - **Hybrid Search**: The key to the RAG agent's performance is its hybrid retrieval strategy, which combines the strengths of lexical and semantic search. This is configured in the `create_server` method:
        1.  **Lexical Search (BM25)**: A [`TantivyBM25Factory`](services/document_store.py:112) is used to create a BM25 index. This is a classical information retrieval algorithm that is very effective for matching keywords.
        2.  **Semantic Search (k-NN)**: A [`UsearchKnnFactory`](services/document_store.py:98) is used to create a k-Nearest Neighbors index on the document embeddings. This allows for finding documents that are semantically similar, even if they don't share the same keywords.
    - **[`HybridIndexFactory`](services/document_store.py:121)**: These two factories are combined into a `HybridIndexFactory`. When a query comes in, it is sent to both the BM25 and k-NN indexes, and the results are combined to produce a final, relevance-ranked list of documents.
    - **[`DocumentStore`](services/document_store.py:124)**: This Pathway class takes the raw data, a splitter, and the `HybridIndexFactory` to build the complete, searchable index.
    - **Serving**: The `run_server` method starts a `DocumentStoreServer`, which exposes the document store via an API, allowing the `RagBasedClassifier` in the `agent` service to query it.