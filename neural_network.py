import numpy as np
import pandas as pd
import jieba

# prepare data
df = pd.read_csv('data/data.csv')
data = np.array(df)
np.random.shuffle(data)

test_ratio = 0.2
rows, variables = data[:, :-1].shape
X_train = data[:int(rows * (1 - test_ratio)), :variables]
y_train = data[:int(rows * (1 - test_ratio)), variables]
X_test = data[int(rows * (1 - test_ratio)):, :variables]
y_test = data[int(rows * (1 - test_ratio)):, variables]

# transform format
data = []
words = []
for sen in X_train[:, 0]:
    word = jieba.lcut(sen)
    data.append(word)
    words += word

start_ind = 2
dictionary = dict(zip(set(words), range(start_ind, len(set(words)) + start_ind)))

dictionary['<emp>'] = 0
dictionary['<unk>'] = 1

X_train = np.ones([len(data), max([len(data[i]) for i in range(len(data))])], dtype=int) * dictionary['<emp>']
for i in range(len(data)):
    for j in range(len(data[i])):
        X_train[i][j] = dictionary[data[i][j]]

# init
lr = 0.001
epoch = 150
doc_dim, embedding_dim, hidden_dim, target_dim = X_train.shape[1], 128, 256, 3

W_xe = np.random.rand(len(dictionary), embedding_dim)
W_eh1 = np.random.rand(hidden_dim, doc_dim, embedding_dim)
b_h1 = np.zeros(hidden_dim)
# W_h1h2 = np.random.rand(hidden_dim, hidden_dim)
# b_h2 = np.zeros(hidden_dim)
W_hy = np.random.rand(hidden_dim, target_dim)

dW_xe = np.zeros(W_xe.shape)
dW_eh1 = np.zeros(W_eh1.shape)
db_h1 = np.zeros(b_h1.shape)
# dW_h1h2 = np.zeros(W_h1h2.shape)
# db_h2 = np.zeros(b_h2.shape)
dW_hy = np.zeros(W_hy.shape)

# train
for j in range(epoch):
    loss = 0
    for i in range(len(X_train)):
        e = W_xe[X_train[i]]
        h1 = np.sum(np.multiply(e, W_eh1), axis=(1, 2)) + b_h1
        h1_t = np.tanh(h1)
        # h2 = np.dot(h1_t, W_h1h2) + b_h2
        # h2_t = np.tanh(h2)
        
        y = np.dot(h1_t, W_hy)
        y = np.exp(y)
        y /= sum(y)
        
        loss -= np.log(y[y_train[i]])
        
        y[y_train[i]] -= 1
        dW_hy = np.outer(h1_t, y)
        h = np.dot(W_hy, y)
        W_hy -= lr * dW_hy
        
        # h = 1 - h ** 2
        # db_h2 = h2 - h
        # dW_h1h2 = np.outer(h1_t, (h2 - h))
        # h = np.dot(W_h1h2, h)
        # b_h2 -= lr * db_h2
        # W_h1h2 -= lr * dW_h1h2
        
        h = 1 - h ** 2
        db_h1 = h1 - h
        dW_eh1 = np.outer((h1 - h), e).reshape(dW_eh1.shape)
        b_h1 -= lr * db_h1
        W_eh1 -= lr * dW_eh1
    print(loss)
    print(j)


# test

data = []
for sen in X_test[:, 0]:
    word = jieba.lcut(sen)
    data.append(word)

X_test = np.ones([len(data), doc_dim], dtype=int) * dictionary['<emp>']
for i in range(len(data)):
    for j in range(len(data[i])):
        if data[i][j] in dictionary:
            X_test[i][j] = dictionary[data[i][j]]
        else:
            X_test[i][j] = dictionary['<unk>']

y_pred = []
for i in range(len(X_test)):
    e = W_xe[X_test[i]]
    h1 = np.sum(np.multiply(e, W_eh1), axis=(1, 2)) + b_h1
    h1_t = np.tanh(h1)
    # h2 = np.dot(h1_t, W_h1h2) + b_h2
    # h2_t = np.tanh(h2)
    
    y = np.dot(h1_t, W_hy)
    y = np.exp(y)
    y /= sum(y)
    y_pred.append(np.argsort(y)[-1])

acc = np.sum(y_test == y_pred) / len(y_test)
print('test:'+str(acc))

y_pred = []
for i in range(len(X_train)):
    e = W_xe[X_train[i]]
    h1 = np.sum(np.multiply(e, W_eh1), axis=(1, 2)) + b_h1
    h1_t = np.tanh(h1)
    # h2 = np.dot(h1_t, W_h1h2) + b_h2
    # h2_t = np.tanh(h2)
    
    y = np.dot(h1_t, W_hy)
    y = np.exp(y)
    y /= sum(y)
    y_pred.append(np.argsort(y)[-1])

acc = np.sum(y_train == y_pred) / len(y_train)
print('train:'+str(acc))
