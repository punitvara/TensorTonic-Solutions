import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def split_heads(x, batch_size, seq_len, num_heads, head_dim):
    x = x.reshape(batch_size, seq_len, num_heads, head_dim)
    return x.transpose(0,2,1,3)
    
def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    # Your code here
    batch_size, seq_len, d_model = Q.shape
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
    head_dim = d_model // num_heads

    # 1. Linear projections: (batch_size, seq_len, d_model)
    q = Q @ W_q
    k = K @ W_k
    v = V @ W_v

    # 2. Split into heads: (batch_size, seq_len, d_model)
    #                   → (batch_size, seq_len, num_heads, head_dim)
    #                   → (batch_size, num_heads, seq_len, head_dim)
    def split_heads(x):
        x = x.reshape(batch_size, seq_len, num_heads, head_dim)
        return x.transpose(0, 2, 1, 3)

    q = split_heads(q)
    k = split_heads(k)
    v = split_heads(v)

    # 3. Scaled dot-product attention (broadcasts over batch_size and num_heads)
    #    q, k, v: (batch_size, num_heads, seq_len, head_dim)
    #    k.swapaxes(-2, -1): (batch_size, num_heads, head_dim, seq_len)
    scores = q @ k.swapaxes(-2, -1) / np.sqrt(head_dim)
    # scores: (batch_size, num_heads, seq_len, seq_len)

    attn = softmax(scores, axis=-1)
    # attn: (batch_size, num_heads, seq_len, seq_len)

    out = attn @ v
    # out: (batch_size, num_heads, seq_len, head_dim)

    # 4. Merge heads back: (batch_size, num_heads, seq_len, head_dim)
    #                   → (batch_size, seq_len, num_heads, head_dim)
    #                   → (batch_size, seq_len, d_model)
    out = out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)

    # 5. Final output projection
    return out @ W_o
    
    # batch_size, seq_len, d_model = Q.shape
    # assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
    # Q_proj = Q @ W_q
    # K_proj = K @ W_k
    # V_proj = V @ W_v

    # head_dim = d_model // num_heads

    # q_head = split_heads(Q_proj, batch_size, seq_len, num_heads, head_dim)
    # k_head = split_heads(K_proj, batch_size, seq_len, num_heads, head_dim)
    # v_head = split_heads(V_proj, batch_size, seq_len, num_heads, head_dim)

    # dot_prod = torch.matmul(q_head, k_head.transpose(-2,-1))

    # attn = softmax(dot_prod / math.sqrt(dot_prod))

    # out = attn @ v_head

    # out = out.transpose(0,2,1,3).reshape(batch_size, seq_len, d_model)

    # return out @ W_o
    
    