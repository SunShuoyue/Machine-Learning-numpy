import numpy as np

# prepare data
rows = 1000
test_ratio = 0.2
variables = 3
x1 = np.random.randint(400, 1000, [int(rows / 2), variables]) * 0.01
y1 = np.ones([int(rows / 2)])
data1 = np.c_[x1, y1]
x0 = np.random.randint(100, 600, [int(rows / 2), variables]) * 0.01
y0 = np.zeros([int(rows / 2)])
data0 = np.c_[x0, y0]
data = np.r_[data1, data0]
np.random.shuffle(data)

X_train = data[:int(rows * (1 - test_ratio)), :variables]
y_train = data[:int(rows * (1 - test_ratio)), variables]
X_test = data[int(rows * (1 - test_ratio)):, :variables]
y_test = data[int(rows * (1 - test_ratio)):, variables]

# test
k = 5

y_pred = np.zeros(len(X_test))
for i, test_sample in enumerate(X_test):
    k_sim = np.argsort(np.sqrt(np.sum((test_sample - X_train)**2, axis=1)))[:k]
    y_pred[i] = int(np.argmax(np.bincount([y_train[j] for j in k_sim])))

acc = (len(y_test)-np.count_nonzero(y_pred-y_test))/len(y_test)

print(acc)
