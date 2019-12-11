# from gensim.models.word2vec import Word2Vec
# import pandas as pd
# import jieba
#
# df = pd.read_csv('data/data.csv')
# data = []
# for sen in list(df.sentence):
#     word = jieba.lcut(sen)
#     data.append(word)
#
# model = Word2Vec(data, size=128, min_count=1)
# model.save("data/word2vec.model")
# model = Word2Vec.load("data/word2vec.model")

import pandas as pd
import numpy as np
import jieba

df = pd.read_csv('data/data.csv')
sentences = list(df.sentence)
words = []
for i in range(len(sentences)):
    sentences[i] = jieba.lcut(sentences[i])
    words += sentences[i]

dictionary = dict(zip(set(words), range(len(set(words)))))

window = 2
input = []
target = []
for i in range(len(sentences)):
    for j in range(len(sentences[i])):
        input.append(sentences[i][j])
        target.append(sentences[i][max(0, j - window):min(j + window, len(sentences[i]))])

# init
lr = 0.001
epoch = 50
dict_dim, embedding_dim = len(dictionary), 128

W_xe = np.random.rand(dict_dim, embedding_dim)
W_ey = np.random.rand(embedding_dim, dict_dim)

dW_xe = np.zeros(W_xe.shape)
dW_ey = np.zeros(W_ey.shape)

for j in range(epoch):
    loss = 0
    for i in range(len(input)):
        w2v = W_xe[dictionary[input[i]]]
        y = np.dot(w2v, W_ey)
        y_pred = np.exp(y)
        y_pred /= sum(y_pred)
        
        loss += -np.sum([y[dictionary[word]] for word in target[i]]) + len(target[i]) * np.log(np.sum(np.exp(y)))
        
        y_pred = y_pred * len(target[i])
        for word in target[i]:
            y_pred[dictionary[word]] -= 1
        
        dW_ey = np.outer(w2v, y_pred)
        W_ey -= lr * dW_ey
        W_xe -= lr * w2v
    print(loss)
    print(j)
