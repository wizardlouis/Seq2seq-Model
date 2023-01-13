# -*- codeing = utf-8 -*-
# @time:2021/12/7 下午2:09
# Author:Xuewen Shen
# @File:Embedding_Dict.py
# @Software:PyCharm

import math

# Input channel Vector Embedding

Emb_n6d2 = [[math.cos(math.pi * 2 / 3), math.sin(math.pi * 2 / 3)],
            [math.cos(math.pi / 3), math.sin(math.pi / 3)],
            [math.cos(0), math.sin(0)],
            [math.cos(-math.pi / 3), math.sin(-math.pi / 3)],
            [math.cos(-math.pi * 2 / 3), math.sin(-math.pi * 2 / 3)],
            [math.cos(math.pi), math.sin(math.pi)]
            ]
Emb_n4d2 = [
    [1., 0.],
    [0., 1.],
    [-1., 0.],
    [0., -1.]
]

Emb_n2d1 = [[1.],
            [-1.]]

# Time period Embedding
Emb_standard = dict(
    t_on=8, Dt_on=4, t_off=10, Dt_off=0, t_delay=75, dt=10, tau=100, t_upb=[87, 109, 131]
)

Emb_long_interval = dict(
    t_on=8, Dt_on=4, t_off=40, Dt_off=0, t_delay=75, dt=10, tau=100, t_upb=[87, 139, 191]
)

Emb_li_d150 = dict(
    t_on=8, Dt_on=4, t_off=40, Dt_off=0, t_delay=150, dt=10, tau=100, t_upb=[162, 214, 266]
)
