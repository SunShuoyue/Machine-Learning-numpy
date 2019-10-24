import numpy as np

# prepare data
rows = 1000
test_ratio = 0.2
variables = 3
x1 = np.random.randint(100, 1000, [int(rows / 2), variables]) * 0.01
y1 = np.ones([int(rows / 2)])
data1 = np.c_[x1, y1]
x0 = np.random.randint(400, 500, [int(rows / 2), variables]) * 0.01
y0 = np.zeros([int(rows / 2)])
data0 = np.c_[x0, y0]
data = np.r_[data1, data0]
np.random.shuffle(data)

X_train = data[:int(rows * (1 - test_ratio)), :variables]
y_train = data[:int(rows * (1 - test_ratio)), variables]
X_test = data[int(rows * (1 - test_ratio)):, :variables]
y_test = data[int(rows * (1 - test_ratio)):, variables]

# train
Xy = np.c_[X_train, y_train]

min_samples_split = None
max_depth = None
min_info_gain = 10e-5


def entropy(y):
    labels = np.unique(y)
    entr = 0
    for label in labels:
        p = np.sum(y == label) / len(y)
        entr += -p * (np.log(p) / np.log(2))
    return entr


def add_layer(Xy):
    largest_impurity = 0
    X = Xy[:, :Xy.shape[1] - 1]
    y = Xy[:, -1]
    for i in range(Xy.shape[1] - 1):
        unique_values = np.unique(X[:, i])
        if len(unique_values) == 1:
            continue
        for value in unique_values:
            if isinstance(value, int) or isinstance(value, float):
                Xy1 = Xy[Xy[:, i] >= value]
                Xy2 = Xy[Xy[:, i] < value]
            else:
                Xy1 = Xy[Xy[:, i] == value]
                Xy2 = Xy[Xy[:, i] != value]
            if Xy1.any() and Xy2.any():
                y1 = Xy1[:, -1]
                y2 = Xy2[:, -1]
                info_gain = entropy(y) - (len(y1) / len(y)) * entropy(y1) - (len(y2) / len(y)) * entropy(y2)
                if info_gain > largest_impurity:
                    largest_impurity = info_gain
                    result = {'feature': i, 'threshold': value, 'left': Xy1, 'right': Xy2}
    if largest_impurity == 0:
        print(0)
        values, counts = np.unique(Xy[:, -1], return_counts=True)
        return values[np.argmax(counts)]
    elif largest_impurity < min_info_gain:
        print(1)
        values, counts = np.unique(result['left'][:, -1], return_counts=True)
        result['left'] = values[np.argmax(counts)]
        values, counts = np.unique(result['right'][:, -1], return_counts=True)
        result['right'] = values[np.argmax(counts)]
        return result
    result['left'] = add_layer(result['left'])
    result['right'] = add_layer(result['right'])
    return result


tree = add_layer(Xy)


# test
def next_layer(tree, sample):
    if isinstance(sample[tree['feature']], int) or isinstance(sample[tree['feature']], float):
        if sample[tree['feature']] >= tree['threshold']:
            if isinstance(tree['left'], dict):
                return next_layer(tree['left'], sample)
            else:
                return tree['left']
        else:
            if isinstance(tree['right'], dict):
                return next_layer(tree['right'], sample)
            else:
                return tree['right']
    else:
        if sample[tree['feature']] == tree['threshold']:
            if isinstance(tree['left'], dict):
                return next_layer(tree['left'], sample)
            else:
                return tree['left']
        else:
            if isinstance(tree['right'], dict):
                return next_layer(tree['right'], sample)
            else:
                return tree['right']


y_pred = []
for sample in X_test:
    y_pred.append(next_layer(tree, sample))

acc = np.sum(y_test == y_pred) / len(y_test)
print(acc)
