import numpy as np

def print_bin(a, end="\n", include_0b=False):
    assert np.issubdtype(a.dtype, np.unsignedinteger)

    # 비트 수 계산 (8, 16, 32, 64 등)
    bits = a.dtype.itemsize * 8
    
    def format_4bit(x):
        # 1. 0b를 제외한 순수 바이너리 문자열 생성 (0 패딩 포함)
        bin_str = format(x, f'0{bits}b')
        # 2. 4글자마다 공백 삽입
        parts = [bin_str[i:i+4] for i in range(0, len(bin_str), 4)]
        # 3. 0b 접두사와 함께 합치기
        if(include_0b):
            return '0b' + '_'.join(parts)
        else:
            return '_'.join(parts)

    # numpy 출력 옵션 적용
    print(np.array2string(a, formatter={'int': format_4bit}), end=end)

def print_non_zero(w):
    # w_uint8 = w.view(np.uint8)
    pos = np.where(w)
    pos_max_len = [len(str(p.max())) for p in pos]

    for i in range(len(pos[0])):
        pos_tuple = tuple([p[i] for p in pos])
        pos_str_list = [f"{p[i]:>{lenp}}" for p, lenp in zip(pos, pos_max_len)]
        pos_str = ", ".join(pos_str_list)
        pos_str = f"({pos_str})"

        print(pos_str, end=" ")
        print_bin(w[pos_tuple])

        
def _get_l_size(
    k: int,
    group_size: int,
    need_dequant: bool = True,
) -> int:
    LUT_G = 4
    LUT_SIZE = 16
    ACT_GROUP_SIZE = 256
    # float16
    x_size = k if need_dequant else 0
    # int16
    l_size = k // LUT_G * LUT_SIZE
    # float32
    ls_size = 1 if (ACT_GROUP_SIZE == -1) else (k // ACT_GROUP_SIZE)
    # float32
    lb_size = 1 if (group_size == 0) else (k // group_size)
    return x_size * 2 + l_size * 2 + max(ls_size * 4, 128) + max(lb_size * 4, 128)