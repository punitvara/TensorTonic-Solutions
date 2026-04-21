import torch
import torch.nn as nn
import math

def create_embedding_layer(vocab_size: int, d_model: int) -> nn.Embedding:
    """
    Create an embedding layer.
    """
    embedding = nn.Embedding(vocab_size, d_model)
    return embedding

def embed_tokens(embedding: nn.Embedding, tokens: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    Convert token indices to scaled embeddings.
    """
    embeddings = embedding(tokens)
    scaled_embeddings = embeddings * math.sqrt(d_model)
    return scaled_embeddings
