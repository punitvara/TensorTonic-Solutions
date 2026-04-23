import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    # Your code here
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)

    x_hat = (x-mean) / np.sqrt(var + eps)
    return gamma * x_hat + beta

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    # Your code here
    batch_size, seq_len, d_model = Q.shape

    assert d_model % num_heads == 0, "d_model should be divisible by num_heads"

    Q_proj = Q @ W_q
    K_proj = K @ W_k
    V_proj = V @ W_v

    head_dim = d_model // num_heads 

    def split_shape(x):
        x = x.reshape(batch_size, seq_len, num_heads, head_dim)
        return x.transpose(0,2,1,3)
    
    q_head = split_shape(Q_proj)
    k_head = split_shape(K_proj)
    v_head = split_shape(V_proj)

    scores = q_head @ k_head.swapaxes(-2,-1) / np.sqrt(head_dim)

    attn = softmax(scores,axis=-1) 

    out = attn @ v_head

    out = out.transpose(0,2,1,3).reshape(batch_size, seq_len, d_model)
    
    return out @ W_o

    
def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    # Your code here
    hidden = np.maximum(0, x@W1 + b1)
    return hidden @ W2 + b2

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    #self attention layer
    attn_out = multi_head_attention(x,x,x,W_q,W_k,W_v,W_o, num_heads)
    h1 = layer_norm(x + attn_out, gamma1, beta1)

    ffn_out = feed_forward(h1, W1, b1, W2, b2)
    h2 = layer_norm(h1 + ffn_out, gamma2, beta2)
    return h2