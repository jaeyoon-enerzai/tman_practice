import numpy as np
from typing import Dict, Optional, Tuple

def hvx_preprocess_weights(
    w: np.ndarray,
    scales: np.ndarray,
    zeros: Optional[np.ndarray] = None,
    bits: int = 4,
    g: int = 4,
    tile_p: int = 512,
    tile_q: int = 64,
    vec_p: int = 128,
    vec_q: int = 4,
    vec_c: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:

    assert w.dtype == "uint8"
    assert scales.dtype == "float16" or scales.dtype == "float32" or scales.dtype == "bfloat16"
    if scales.dtype != "float16":
        scales = scales.astype("float16")
        zeros = zeros.astype("float16") if zeros is not None else None
    # 4 = sizeof(int32/float) / sizeof(uint8)
    assert vec_p // 4 == vec_c
    M, K = w.shape
    assert M >= vec_p, f"out features {M} should be larger than vec_p {vec_p}"

    P = M * bits
    Q = K // g

    # (M, K, bits)
    w = np.stack([(w >> ib) & 1 for ib in range(bits)], axis=-1)
    # (M, K, bits) -> (M, bits, K) -> (M, bits, K) -> (M, bits, K // g, g)
    w = w.transpose(0, 2, 1).reshape(M, bits, Q, g)
    # (M, bits, K // g, g) -> (M, bits, Q)
    w = sum([(w[:, :, :, ig] << ig) for ig in range(g)])
    # (M, bits, Q) -> (M // vec_p, vec_p, bits, Q) -> (M // vec_p, bits, vec_p, Q) -> (P // vec_p, vec_p, Q)
    w = w.reshape(M // vec_p, vec_p, bits, Q).transpose(0, 2, 1, 3)
    # Interleave even and odd vec_c of w_vec
    # 0, 1 -> even bytes of w_vec -> c_vec_0, c_vec_2 -> c_bitsum_lo
    # 2, 3 ->  odd bytes of w_vec -> c_vec_1, c_vec_3 -> c_bitsum_hi
    # w_vec = w0/w2/w0/w2......w1/w3/w1/w3
    # c_vec_0, c_vec_2 = w0/w0......w1/w1
    # c_vec_1, c_vec_3 = w2/w2......w3/w3
    w = w.reshape(P // vec_p, 2, 2, vec_c, Q).transpose(0, 2, 3, 1, 4)
    w = w.reshape(P // tile_p, tile_p, Q // tile_q, tile_q).transpose(0, 2, 1, 3)
    #             0            1            2                3      4                5
    w = w.reshape(P // tile_p, Q // tile_q, tile_p // vec_p, vec_p, tile_q // vec_q, vec_q).transpose(0, 1, 2, 4, 5, 3)
    # Pack and interleave: q = 0 -> w_vec_lo_bo, q = 1 -> w_vec_lo_to, q = 2 -> w_vec_hi_bo, q = 3 -> w_vec_hi_to
    # lo -> low 128 bytes, hi -> high 128 bytes, bo -> bot 4 bit in a byte, to -> top 4 bit in a byte
    w = w.reshape(-1, vec_q, vec_p).reshape(-1, vec_q // 2, 2, vec_p).transpose(0, 1, 3, 2)
    w = sum([(w[:, :, :, n] << (n * g)) for n in range(2)])
    w = w.reshape(P // tile_p, Q // tile_q, tile_p // vec_p, tile_q // vec_q, vec_q // 2, vec_p)
    # Reshape for easy tiling
    w = np.ascontiguousarray(w).view(np.int32).reshape(P // tile_p, -1)

    if scales.size >= M:  # GPTQ
        group_size = K // scales.shape[1] ## group along K axis
        q_group_size = group_size // g
        scales = scales.reshape(P // tile_p, tile_p // bits, Q // tile_q, tile_q // q_group_size).transpose(0, 2, 1, 3)
        #                       0            1            2                        3      4
        scales = scales.reshape(P // tile_p, Q // tile_q, tile_p // bits // vec_p, vec_p, tile_q // q_group_size).transpose(0, 1, 2, 4, 3)
        # s_vec = s0/s0......s1/s1......s2/s2......s3/s3
        # s_vec_lo_lo, s_vec_lo_hi = s0/s0......s1/s1 -> c_vec_0, c_vec_2 -> c_bitsum_lo
        # no need for interleaving
        if zeros is not None:
            zeros = zeros.reshape(P // tile_p, tile_p // bits, Q // tile_q, tile_q // q_group_size).transpose(0, 2, 1, 3)
            zeros = zeros.reshape(P // tile_p, Q // tile_q, tile_p // bits // vec_p, vec_p, tile_q // q_group_size).transpose(0, 1, 2, 4, 3)
            # (c * ls + lb) * s + z * s * lb * 2
            # = (c * ls + lb + z * lb * 2) * s
            # = (c * ls + (z * 2 + 1) * lb) * s
            zeros = zeros * 2 + 1
            scales = np.stack([scales, zeros], axis=-2)
        scales = np.ascontiguousarray(scales).view(np.int32).reshape(P // tile_p, -1)
    else:  # BitNet
        scales = scales.view(np.uint16).reshape(1, -1)
        # [ERROR] [Qnn ExecuTorch]: QnnDsp <E> Dma execution failed on the skel side. result = 1100 transport error = 0
        # Padding to vec_p
        # TODO: verify if the padding is needed
        if scales.nbytes < vec_p:
            scales = np.resize(scales, (1, vec_p // np.dtype("int16").itemsize))
    return w, scales


if __name__ == "__main__":
    original_formatter = np.get_printoptions()['formatter'] # Save original formatter
    np.set_printoptions(formatter={'uint8': lambda x: format(x, '#08b')})

    def print2(a):
        # 비트 수 계산 (8, 16, 32, 64 등)
        bits = a.dtype.itemsize * 8
        
        def format_4bit(x):
            # 1. 0b를 제외한 순수 바이너리 문자열 생성 (0 패딩 포함)
            bin_str = format(x, f'0{bits}b')
            # 2. 4글자마다 공백 삽입
            parts = [bin_str[i:i+4] for i in range(0, len(bin_str), 4)]
            # 3. 0b 접두사와 함께 합치기
            return '0b ' + ' '.join(parts)

        # numpy 출력 옵션 적용
        print(np.array2string(a, formatter={'int': format_4bit}))

    # w = np.tile(np.arange(16), (512, 32)).astype(np.uint8)
    # w[0, 0:16] = np.arange(16).astype(np.uint8)

    w_shape = (1024, 1024)
    val_col_test_list = [
        (0, 0),
        (0, 1),
        (0, 8),
        (0, 15),
        (0, 255),
        (0, 256),
        (0, 512),
        
    ]

    val_row_test_list = [
        (0, 0),
        (1, 0),
        (2, 0),
        (63, 0),
        (64, 0),
        (128, 0),
        (256, 0),
        (384, 0),
    ]

    test_dict = {
        "value along col axis": val_col_test_list,
        "value along row axis": val_row_test_list
    }

    
    for key, test_list in test_dict.items():
        shape_flag = True
        print(key)
        for r, c in test_list:
            w = np.zeros((w_shape), dtype=np.uint8)
            w[r, c] = 15
            s = np.array([0.12312]).astype(np.float16)
            wp, sp = hvx_preprocess_weights(w, s, bits=4, tile_p=512, tile_q=64, vec_p=128, vec_c=32, g=4)
            wp_uint8 = wp.view(np.uint8)

            if shape_flag:
                print(f"w shape: {w.shape}")
                print(f"wp_uint8 shape: {wp_uint8.shape}")
                shape_flag = False
            rows, cols = np.where(wp_uint8)
            print(f"## w[{r}, {c}] = 15")
            for row, col in zip(rows, cols):
                print("({:>5}, {:>5})".format(row, col), end=" ")
                print2(wp_uint8[row, col])
            print("\n")

