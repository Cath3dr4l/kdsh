# ARC2 Embedder Service

This directory contains the sentence embedding service for the **ARC2 (Agentic AI Research Review and Conference Classification)** project. It is a standalone service deployed on [Modal](https://modal.com/) to provide on-demand text embeddings.

## 🚀 Overview

The Embedder Service is a crucial piece of infrastructure that serves a single purpose: to convert text into high-quality numerical vectors (embeddings). These embeddings are used by both the **Indexer Service** to build its semantic search index and the **Agent Service** for various similarity-based tasks.

By deploying this as a separate, serverless function on Modal, we ensure:
- **Scalability**: The service can handle high-throughput requests without manual intervention.
- **Efficiency**: We leverage powerful hardware (including GPUs) on Modal for fast inference.
- **Decoupling**: The core application logic is decoupled from the embedding model, making it easy to update or swap models in the future.

## 🛠️ Tech Stack

-   Python 3.11
-   [Modal](https://modal.com/)
-   [Sentence Transformers](https://www.sbert.net/)

### Embedding Model
The service uses the `dunzhang/stella_en_1.5B_v5` model, a powerful sentence transformer known for its performance on diverse English-language tasks.

## ⚙️ Setup and Execution

### Prerequisites

-   Python 3.11+
-   A [Modal](https://modal.com/) account and the Modal CLI installed and configured.

### Installation & Deployment

1.  Navigate to the `embedder` directory:
    ```bash
    cd embedder
    ```
2.  The dependencies are defined directly in the `modal_service.py` script and will be installed by Modal in the container environment.
3.  Deploy the service to Modal:
    ```bash
    modal deploy modal_service.py
    ```

### Using the Service

Once deployed, Modal provides a unique URL for the service. The [Agent](../agent/README.md) and [Indexer](../indexer/README.md) services are configured to call this endpoint to get embeddings.

To run the service locally for testing, you can use the following command:
```bash
modal serve modal_service.py
```

## 🏛️ Architecture Deep Dive

The entire service is defined in [`modal_service.py`](modal_service.py:1).

- **Modal Configuration**:
    - **`volume`**: A `modal.Volume` is created to act as a persistent, shared file system. This is used to cache the downloaded sentence transformer model, so it doesn't need to be re-downloaded every time the service starts.
    - **`image`**: A `modal.Image` defines the container environment. It specifies the base image (`debian_slim`) and the required Python packages (`sentence_transformers`, `xformers`).
    - **`app`**: A `modal.App` ties the image and configuration together.

- **Model Loading and Caching**:
    - The [`save_model`](modal_service.py:30) function is a one-time setup step. It downloads the `dunzhang/stella_en_1.5B_v5` model and saves it to the persistent `modal.Volume`. This function only needs to be run once.
    - The [`Model` class](modal_service.py:41) is the main application class.
        - The `@modal.enter()` decorator on the `setup` method ensures that the model is loaded from the volume into GPU memory when the container starts. This "warm-up" process means the model is ready to go before the first request arrives, minimizing latency.

- **Inference Endpoint**:
    - The [`inference`](modal_service.py:56) method is the core of the service. It takes a text prompt, runs it through the loaded sentence transformer model, and returns the resulting embedding vector.
    - The `@modal.method()` decorator exposes this class method as a callable endpoint.

- **API Serving**:
    - The `@app.local_entrypoint()` function defines how the service is run. It sets up a FastAPI application that wraps the Modal `Model` class.
    - It exposes a `/embeddings` endpoint that takes a text query, calls the `myModal.inference.remote.aio()` method to get the embedding, and returns it as a JSON response. This provides a standard, easy-to-use interface for other services in the ARC2 ecosystem.