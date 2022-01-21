# -*- codeing = utf-8 -*-
# @time:2021/12/7 下午2:09
# Author:Xuewen Shen
# @File:Embedding_Dict.py
# @Software:PyCharm

import math

Emb_n6d2 = [[math.cos(math.pi * 2 / 3), math.sin(math.pi * 2 / 3)],
            [math.cos(math.pi / 3), math.sin(math.pi / 3)],
            [math.cos(0), math.sin(0)],
            [math.cos(-math.pi / 3), math.sin(-math.pi / 3)],
            [math.cos(-math.pi * 2 / 3), math.sin(-math.pi * 2 / 3)],
            [math.cos(math.pi), math.sin(math.pi)]
            ]

Emb_n2d1 = [[1.],
            [-1.]]
