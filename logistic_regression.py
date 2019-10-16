import numpy as np

# prepare data
rows = 1000
test_ratio = 0.2
variables = 2
x1 = np.random.randint(900, 1000, [int(rows / 2), variables]) * 0.01
y1 = np.ones([int(rows / 2)])
data1 = np.c_[x1, y1]
x0 = np.random.randint(100, 200, [int(rows / 2), variables]) * 0.01
y0 = np.zeros([int(rows / 2)])
data0 = np.c_[x0, y0]
data = np.r_[data1, data0]
np.random.shuffle(data)

X_train = data[:int(rows * (1 - test_ratio)), :variables]
y_train = data[:int(rows * (1 - test_ratio)), variables]
X_test = data[int(rows * (1 - test_ratio)):, :variables]
y_test = data[int(rows * (1 - test_ratio)):, variables]

# train
lr = 0.01
epoch = 100

W = np.random.random([X_train.shape[1]])
b = np.random.random()

for i in range(epoch):
    z = np.dot(X_train, W) + b
    h = 1 / (1 + np.exp(-z))
    loss = (-y_train * np.log(h) - (1 - y_train) * np.log(1 - h)).mean()
    
    dw = np.dot(X_train.T, (h - y_train)) / y_train.shape[0]
    db = (h - y_train).mean()
    W -= lr * dw
    b -= lr * db
    
    if i % 500 == 0:
        print(f'loss: {loss}')

# test
z = np.dot(X_test, W) + b
h = 1 / (1 + np.exp(-z)) > 0.5
acc = sum(h == y_test) / y_test.shape[0]
