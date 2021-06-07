"""
@Time    : 2020/12/28 21:16
@Author  : Affreden
@Email   : affreden@gmail.com
@File    : num_predict.py
"""
import torch
import numpy as np
import random as rd
import time


def onehot_encode(i):
    label = np.zeros(shape=9)
    label[i - 1] = 1
    return label
    pass


NUM_DIGITS = 10
EPOCH = 100
BATCH_SIZE = 128

now = time.time()
rd.seed(time)
train_x = torch.Tensor([onehot_encode(rd.randint(1, 9)) for i in range(1024)])
rd.seed(time)
train_y = torch.LongTensor([rd.randint(0, 8) for i in range(1024)])

model = torch.nn.Sequential(
    torch.nn.Linear(9, 400),
    torch.nn.ReLU(),
    torch.nn.Linear(400, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 9)
)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
for epoch in range(EPOCH):
    for start in range(0, len(train_x), BATCH_SIZE):
        end = start + BATCH_SIZE
        batchX = train_x[start: end]
        batchY = train_y[start: end]
        y_pred = model(batchX)
        loss = loss_fn(y_pred, batchY)
        print("Epoch", epoch, loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

rd.seed(1)
test_x = torch.Tensor([onehot_encode(rd.randint(1, 9)) for i in range(100)])
rd.seed(1)
test_y = torch.Tensor([rd.randint(0, 8) for i in range(100)])

with torch.no_grad():
    test_Y = model(test_x)

test_Y = test_Y.max(1)[1].cpu().tolist()

print(test_Y)
print(test_y.int().tolist())
