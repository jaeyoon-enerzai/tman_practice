# save_test_data.py
import numpy as np
from preprocess_weights import hvx_preprocess_weights


if __name__ == "__main__":
    M = 1536
    K = 1024
    N = 1024
    bits = 4
    tile_p = 512
    tile_q = 64
    vec_p = 128
    vec_q = 4
    vec_c = vec_p // 4
    g = 4
    group_size = 128
    P = M * bits
    Q = K // g
    tile_M = tile_p // bits
    tile_K = tile_q * g
    
    # Generate random weights
    w_org = np.random.rand(M*K).astype(np.float16).reshape(M, K)
    scale = w_org.reshape(M, K // group_size, group_size).max(axis=-1) / (2**bits - 1)
    w = np.clip(np.round(w_org.reshape(M, K // group_size, group_size) / scale[:, :, np.newaxis]).astype(np.uint8), 0, 15).reshape(M, K)
    w_deq = (w.reshape(M, K // group_size, group_size) * scale[:, :, np.newaxis]).reshape(M, K)
    
    # Create dequantization LUT
    wdeq_lut = (np.arange(64) % (2**bits)).reshape(1, 1, 1, 4, 16) * scale.reshape(M // tile_M, tile_M // 4, 4, K // group_size).transpose(0, 3, 1, 2)[:, :, :, :, np.newaxis]
    wdeq_lut = wdeq_lut.astype(np.float16).reshape(M // tile_M, -1)
    
    # Preprocess weights
    wp, sp = hvx_preprocess_weights(w, scale, bits=bits, tile_p=tile_p, tile_q=tile_q, vec_p=vec_p, vec_c=vec_c, g=g)
    wp = wp.view(np.uint8).reshape((P // tile_p, Q // tile_q, tile_p // vec_p, tile_q // vec_q, vec_q // 2, vec_p))
    wp = wp.reshape([P // tile_p, -1])
    
    # Save test data
    print(f"Saving test data...")
    print(f"wp shape: {wp.shape}, dtype: {wp.dtype}")
    print(f"wdeq_lut shape: {wdeq_lut.shape}, dtype: {wdeq_lut.dtype}")
    print(f"w_deq shape: {w_deq.shape}, dtype: {w_deq.dtype}")
    
    # Save to binary files
    wp.tofile('../cpp/data/wp.bin')
    wdeq_lut.tofile('../cpp/data/wdeq_lut.bin')
    w_deq.tofile('../cpp/data/w_deq_ref.bin')
    
    # Save metadata
    metadata = {
        'M': M,
        'K': K,
        'N': N,
        'bits': bits,
        'tile_p': tile_p,
        'tile_q': tile_q,
        'vec_p': vec_p,
        'vec_q': vec_q,
        'g': g,
        'group_size': group_size,
        'tile_M': tile_M,
        'tile_K': tile_K,
        'P': P,
        'Q': Q,
        'wp_shape': wp.shape,
        'wdeq_lut_shape': wdeq_lut.shape,
    }
    
    import json
    with open('../data/metadata.json', 'w') as f:
        json.dump({k: int(v) if isinstance(v, (np.integer, int)) else 
                   [int(x) for x in v] if isinstance(v, (tuple, list)) else v 
                   for k, v in metadata.items()}, f, indent=2)
    
    print(f"Data saved successfully!")
    print(f"  wp.bin: {wp.nbytes / 1024:.2f} KB")
    print(f"  wdeq_lut.bin: {wdeq_lut.nbytes / 1024:.2f} KB")
    print(f"  w_deq_ref.bin: {w_deq.nbytes / 1024:.2f} KB")