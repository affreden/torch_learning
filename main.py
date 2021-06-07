import torch
import numpy as np

NUM_DIGITS = 10


def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)][::-1])


def fizz_buzz_encode(i):
    if i % 15 == 0:
        return 3
    elif i % 5 == 0:
        return 2
    elif i % 3 == 0:
        return 1
    else:
        return 0


def fizz_buzz_decode(i, prediction):
    return [str(i), "fizz", "buzz", "fizz_buzz"][prediction]


def helper(i):
    print(fizz_buzz_decode(i, fizz_buzz_encode(i)))


# for i in range(1,20):
#     helper(i)
trX = torch.Tensor([binary_encode(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)])
trY = torch.LongTensor([fizz_buzz_encode(i) for i in range(101, 2 ** NUM_DIGITS)])

NUM_HIDDEN1 = 500
NUM_HIDDEN2 = 100
model = torch.nn.Sequential(
    torch.nn.Linear(NUM_DIGITS, NUM_HIDDEN1),
    torch.nn.ReLU(),
    torch.nn.Linear(NUM_HIDDEN1, NUM_HIDDEN2),
    # torch.nn.AvgPool2d((2,2),1),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(NUM_HIDDEN2, 4)

)

if torch.cuda.is_available():
    model = model.cuda(1)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

BATCH_SIZE = 128
for epoch in range(500):
    for start in range(0, len(trX), BATCH_SIZE):
        end = start + BATCH_SIZE
        batchX = trX[start: end]
        batchY = trY[start: end]
        if torch.cuda.is_available():
            batchX = batchX.cuda(1)
            batchY = batchY.cuda(1)
        y_pred = model(batchX)
        loss = loss_fn(y_pred, batchY)
        print("Epoch", epoch, loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


test_X = torch.Tensor([binary_encode(i, NUM_DIGITS) for i in range(1, 101)])
if torch.cuda.is_available():
    test_X = test_X.cuda(1)
with torch.no_grad():
    test_Y = model(test_X)

test_Y = test_Y.max(1)[1].cpu().tolist()
predictions = zip(range(1, 101), test_Y)
print([fizz_buzz_decode(i, x) for i, x in predictions])
sum = 0
for i in range(1, 101):
    if test_Y[i - 1] == fizz_buzz_encode(i):
        sum += 1
print(sum)
