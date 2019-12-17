import numpy as np
import pickle


def train_test_split(X, y, test_ratio=0.3):
    data = np.array(X)
    label = np.array(y)
    l = list(range(len(label)))
    np.random.shuffle(l)
    test_index = l[:int(len(l) * test_ratio)]
    train_index = l[int(len(l) * test_ratio):]
    return data[train_index], label[train_index], data[test_index], label[test_index]


# prepare data
with open('data/digit.pkl', 'rb') as f:
    data = pickle.load(f)

X = np.array(data['X'])
y = np.array(data['y'])

digit1 = 1
digit2 = 8
idx = np.append(np.where(y == digit1)[0], np.where(y == digit2)[0])
y = y[idx]
y[y == digit1] = -1
y[y == digit2] = 1
X = X[idx]
X_train, y_train, X_test, y_test = train_test_split(X, y)

# train
number_of_classifiers = 5

n_samples, n_features = np.shape(X_train)
w = np.full(n_samples, (1 / n_samples))
clfs = []
for _ in range(number_of_classifiers):
    min_error = float('inf')
    for feature_i in range(n_features):
        unique_values = np.unique(X_train[:, feature_i])
        for value in unique_values:
            p = 1
            prediction = np.ones(np.shape(y_train))
            prediction[X_train[:, feature_i] < value] = -1
            error = sum(w[y_train != prediction])
            if error > 0.5:
                error = 1 - error
                p = -1
            if error < min_error:
                polarity = p
                threshold = value
                feature_index = feature_i
                min_error = error
    alpha = 0.5 * np.log((1.0 - min_error) / (min_error + 1e-10))
    predictions = np.ones(np.shape(y_train))
    negative_idx = (polarity * X_train[:, feature_index] < polarity * threshold)
    predictions[negative_idx] = -1
    w *= np.exp(-alpha * y_train * predictions)
    w /= np.sum(w)
    
    clfs.append({'feature_index': feature_index, 'threshold': threshold, 'polarity': polarity, 'alpha': alpha})

# test
n_samples = np.shape(X_test)[0]
y_pred = np.zeros((n_samples, 1))
for clf in clfs:
    predictions = np.ones(np.shape(y_pred))
    negative_idx = (clf['polarity'] * X_test[:, clf['feature_index']] < clf['polarity'] * clf['threshold'])
    predictions[negative_idx] = -1
    y_pred += clf['alpha'] * predictions

y_pred = np.sign(y_pred).flatten()

acc = np.sum(y_test == y_pred) / len(y_test)
print(acc)
