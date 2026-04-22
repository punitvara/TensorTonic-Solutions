import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    PE = np.zeros((seq_length, d_model))
    for pos in range(seq_length):
        for i in range(0, d_model, 2):          # step by 2: handle pair (2i, 2i+1)
            denom = 10000 ** (i / d_model)
            PE[pos, i]     = np.sin(pos / denom)
            if i + 1 < d_model:                 # guard for odd d_model
                PE[pos, i + 1] = np.cos(pos / denom)
    return PE