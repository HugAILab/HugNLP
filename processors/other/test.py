import numpy as np


def fun(token_type_id):
    if np.sum(np.array(token_type_id)) == 0:  # 说明只有一个segment
        return None
    seg1_start = 1
    seg1_end = -1
    seg2_start = -1
    seg2_end = -1
    for ei, token_type in enumerate(token_type_id):
        if token_type == 1 and seg1_end == -1:
            seg1_end = ei - 2
            seg2_start = ei
            continue
        if token_type == 0 and seg1_end != -1:
            seg2_end = ei - 2
            break
    print('seg1_start=', seg1_start)
    print('seg1_end=', seg1_end)
    print('seg2_start=', seg2_start)
    print('seg2_end=', seg2_end)


if __name__ == '__main__':
    token_type_id = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
    fun(token_type_id)
