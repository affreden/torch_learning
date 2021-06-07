"""
@Time    : 2020/12/29 10:35
@Author  : Affreden
@Email   : affreden@gmail.com
@File    : test.py
"""
import torch
import random as rd
import time
import numpy as np

time = time.time()
rd.seed(1)
temp1 = torch.tensor([rd.randint(0, 9) for i in range(48)])
torch.seed()
temp2 = torch.randn(2, 3, 4)
print(temp2.tolist())
temp3 = temp2.permute(2, 1, 0)
print(temp3.tolist())
temp1 = torch.tensor([rd.randint(0, 9) for i in range(24)])
torch.seed()
temp2 = torch.randn(2, 3, 4)
print(temp2.tolist())
temp3=temp2.permute(0, 2, 1)
print(temp3.tolist())