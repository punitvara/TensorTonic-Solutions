import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    # Your code here
    # QKV shape (batch_size, seq_len, head_dim) -2,-1 meaning last swapping for K transpose
    dot_prod = torch.matmul(Q, K.transpose(-2,-1))
    # last is dimension of model in Q
    scaled_dot_product = dot_prod / math.sqrt(Q.size(-1))
    # softmax across col values so row total would be 1. each row represent query and col represent key
    attention = F.softmax(scaled_dot_product, dim=-1) @ V
    return attention