"""
@Time    : 2020/12/26 11:54
@Author  : Affreden
@Email   : affreden@gmail.com
@File    : my_model.py
"""

import torch.nn as tn


class My_model:
    def __init__(self):
        self.model = tn.Sequential(
            tn.Conv2d()
        )
