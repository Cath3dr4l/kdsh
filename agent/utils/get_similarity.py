import numpy as np
import utils.embedding_client as embedding_client


def get_similarity(vector1, vector2):
    # Convert inputs to numpy arrays
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)

    # Calculate dot product
    dot_product = np.dot(vector1, vector2)

    # Calculate magnitudes
    magnitude1 = np.sqrt(np.sum(vector1 * vector1))
    magnitude2 = np.sqrt(np.sum(vector2 * vector2))

    # Avoid division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    # Calculate cosine similarity
    similarity = dot_product / (magnitude1 * magnitude2)

    return float(similarity)


if __name__ == "__main__":
    # Example usage
    text1 = "This is a sample text"
    text2 = "Dogs are cute animals"
    client = embedding_client.EmbeddingClient()
    vector1 = client.embed(text1)
    vector2 = client.embed(text2)
    similarity = get_similarity(vector1, vector2)
    print(f"Cosine similarity: {similarity}")
