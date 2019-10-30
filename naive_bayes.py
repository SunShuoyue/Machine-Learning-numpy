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

# train
data = []
words = []
for sen in X_train[:, 0]:
    word = jieba.lcut(sen)
    data.append(word)
    words += word

dictionary = dict(zip(set(words), range(len(set(words)))))
classes = np.unique(df.label)
y_train = np.array(df.label)
bayes = np.ones([len(classes), len(dictionary)]) * 10e-12
for i in range(len(data)):
    length = len(data[i])
    for word in data[i]:
        bayes[y_train[i]][dictionary[word]] += 1 / length

bayes = bayes * (np.bincount(np.array(y_train, dtype=int)) / len(y_train)).reshape(-1, 1)

# test
y_pred = []
for sen in X_test[:, 0]:
    words = jieba.lcut(sen)
    pred = [np.nan] * len(classes)
    for i in classes:
        lik = 0
        for word in words:
            if word in dictionary:
                lik += np.log(bayes[i][dictionary[word]])
        pred[i] = lik
    y_pred.append(np.argsort(pred)[-1])

acc = np.sum(y_test == y_pred) / len(y_test)
print(acc)
