import numpy as np
import csv
train_X = np.genfromtxt('train_X_de.csv', dtype=np.float64, delimiter=',', skip_header=1)
train_Y = np.genfromtxt('train_Y_de.csv', dtype=np.float64, delimiter=',', skip_header=0)


def split_data_set(data_X, data_Y, feature_index, threshold):
    left_X = []
    right_X = []
    left_Y = []
    right_Y = []
    for i in range(len(data_X)):
        if data_X[i][feature_index] < threshold:
            left_X.append(data_X[i])
            left_Y.append(data_Y[i])
        else:
            right_X.append(data_X[i])
            right_Y.append(data_Y[i])

    return left_X, left_Y, right_X, right_Y


def calculate_gini_index(Y_subsets):
    gini_index = 0
    total_instances = sum(len(Y) for Y in Y_subsets)
    classes = sorted(set([j for i in Y_subsets for j in i]))

    for Y in Y_subsets:
        m = len(Y)
        if m == 0:
            continue
        count = [Y.count(c) for c in classes]
        gini = 1.0 - sum((n / m) ** 2 for n in count)
        gini_index += (m / total_instances) * gini

    return gini_index


def get_best_split(X, Y):
    X = np.array(X)
    best_gini_index = 9999
    best_feature = 0
    best_threshold = 0
    for i in range(len(X[0])):
        thresholds = set(sorted(X[:, i]))
        for t in thresholds:
            left_X, left_Y, right_X, right_Y = split_data_set(X, Y, i, t)
            if len(left_X) == 0 and len(right_X) == 0:
                continue
            gini_index = calculate_gini_index([left_Y, right_Y])
            if gini_index < best_gini_index:
                best_gini_index, best_feature, best_threshold = gini_index, i, t
    return best_feature, best_threshold


class Node:
    def __init__(self, predicted_class, depth):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.depth = depth
        self.left = None
        self.right = None


def construct_tree(X, Y, max_depth, min_size, depth):
    classes = list(set(Y))
    predicted_class = classes[np.argmax([np.sum(Y == c) for c in classes])]
    node = Node(predicted_class, depth)

    # check is pure
    if len(set(Y)) == 1:
        return node

    # check max depth reached
    if depth >= max_depth:
        return node

    # check min subset at node
    if len(Y) <= min_size:
        return node

    feature_index, threshold = get_best_split(X, Y)

    if feature_index is None or threshold is None:
        return node

    node.feature_index = feature_index
    node.threshold = threshold

    left_X, left_Y, right_X, right_Y = split_data_set(X, Y, feature_index, threshold)

    node.left = construct_tree(np.array(left_X), np.array(left_Y), max_depth, min_size, depth + 1)
    node.right = construct_tree(np.array(right_X), np.array(right_Y), max_depth, min_size, depth + 1)
    return node


def predict(root, X):
    node = root
    while node.left:
        if X[node.feature_index] < node.threshold:
            node = node.left
        else:
            node = node.right
    return node.predicted_class
st=Node(0,0)
# print("OMG")
st=construct_tree(train_X[:int(0.7*len(train_X))],train_Y,10000,1,0)
# print(OMG)
if __name__ == "__main__":
    import sys
    c = sys.argv[1]
    for i in range(2, len(sys.argv)):
        c += ' ' + sys.argv[i]
    train_data = np.genfromtxt(c, dtype=np.float64, delimiter=',', skip_header=1)
    # result_data = classify_points_using_knn(train_X, train_Y, train_data, n_ln_norm, k)
    rst = {}
    with open('predicted_test_Y_de.csv', 'w') as g:
        writer = csv.DictWriter(g, fieldnames=['t'])
        for i in range(len(train_data)):
            rst['t'] = int(predict(st,train_data[i]))
            # print(train_data[i], "OMG!!")
            writer.writerow(rst)