import numpy as np

# prepare data
rows = 1000
test_ratio = 0.2
variables = 2
w1, w2, b = 5, 8, 6
X = np.random.randint(1,1000,rows*2)/100
X = X.reshape(rows,-1)
y = w1*X[:,0] + w2*X[:,1] + b + np.random.random(rows)
data = np.c_[X, y]
np.random.shuffle(data)

X_train = data[:int(rows * (1 - test_ratio)), :variables]
y_train = data[:int(rows * (1 - test_ratio)), variables]
X_test = data[int(rows * (1 - test_ratio)):, :variables]
y_test = data[int(rows * (1 - test_ratio)):, variables]

# train
lr = 0.001
epoch = 20

W = np.random.random([X_train.shape[1]])
b = np.random.random()

for i in range(epoch):
    loss = 0
    for j in range(len(y_train)):
        h = np.dot(X_train[j], W) + b
        loss += (h - y_train[j])**2
        
        dw = np.dot(X_train[j], 2*(h - y_train[j]))
        db = 2 * (h - y_train[j])
        W -= lr * dw
        b -= lr * db
    
    if i % 1 == 0:
        print(f'loss: {loss/y_train.shape[0]}')
