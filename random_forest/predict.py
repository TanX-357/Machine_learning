import random
import pandas as pd
import math
import numpy as np
import csv

t_data=pd.read_csv("train_X_rf.csv")
y = np.genfromtxt('train_Y_rf.csv', dtype=np.float64, delimiter=',', skip_header=0)
t_data['out']=y #mergigng the input and output training dataset as my implementation requires that
train_XY=np.array(t_data)
class Node:
    def __init__(self, predicted_class, depth):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.depth = depth
        self.left = None
        self.right = None


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


def get_best_split(X, Y, actual_no_of_features):
    X = np.array(X)
    total_no_of_features = len(X[0])
    features_list = []
    while len(features_list) < actual_no_of_features:
        random_no = random.randint(0, total_no_of_features - 1)
        if random_no not in features_list:
            features_list.append(random_no)
    best_gini_index = 9999
    best_feature = 0
    best_threshold = 0
    for i in features_list:
        thresholds = set(sorted(X[:, i]))
        for t in thresholds:
            left_X, left_Y, right_X, right_Y = split_data_set(X, Y, i, t)
            if len(left_X) == 0 and len(right_X) == 0:
                continue
            gini_index = calculate_gini_index([left_Y, right_Y])
            if gini_index < best_gini_index:
                best_gini_index, best_feature, best_threshold = gini_index, i, t
    return best_feature, best_threshold


def construct_tree(X, Y, min_size, depth, no_of_randm_features):
    classes = list(set(Y))
    predicted_class = classes[np.argmax([np.sum(Y == c) for c in classes])]
    node = Node(predicted_class, depth)

    # check is pure
    if len(set(Y)) == 1:
        return node

    # check min subset at node
    if len(Y) <= min_size:
        return node

    feature_index, threshold = get_best_split(X, Y, no_of_randm_features)

    if feature_index is None or threshold is None:
        return node

    node.feature_index = feature_index
    node.threshold = threshold

    left_X, left_Y, right_X, right_Y = split_data_set(X, Y, feature_index, threshold)

    node.left = construct_tree(np.array(left_X), np.array(left_Y),min_size, depth + 1, no_of_randm_features)
    node.right = construct_tree(np.array(right_X), np.array(right_Y), min_size, depth + 1,no_of_randm_features)
    return node


def predict(root, X):
    node = root
    while node.left:
        if X[node.feature_index] < node.threshold:
            node = node.left
        else:
            node = node.right
    return node.predicted_class


def get_bootstrap_samples(train_XY, num_bootstrap_samples):
    samples = list()
    random.seed(19)
    for i in range(num_bootstrap_samples):
        sample = list()
        m = len(train_XY)
        while len(sample) < m:
            index = random.randint(0, m - 1)
            sample.append(list(train_XY[index]))
        samples.append(sample)
    return samples


def get_trained_models_using_bagging(train_XY, num_bootstrap_samples, bootstrap_samples):
    models = list()
    random.seed(17)
    for i in range(num_bootstrap_samples):
        model = construct_tree(np.array(bootstrap_samples[i])[:, :-1], np.array(bootstrap_samples[i])[:, -1], 1, 0,
                               math.ceil(math.sqrt(num_bootstrap_samples)))
        models.append(model)
    return models


def predict_using_bagging(models, test_X):
    predictions = [predict(model, test_X) for model in models]
    classes = sorted(list(set(predictions)))
    max_voted_class = -1
    max_votings = -1
    for c in classes:
        if (predictions.count(c) > max_votings):
            max_voted_class = c
            max_votings = predictions.count(c)

    return max_voted_class


def get_out_of_bag_error(models, train_X, bootstrap_samples):
    oob = 0
    num_of_elements_predicted = 0
    train_XY = list(train_X)
    for train_elem in (train_XY):
        train_elem = list(train_elem)
        can_go_in_loop = True
        num_bags_not_having_train_elem = 0
        misclassified_count = 0
        for index in range(len(bootstrap_samples)):
            for i in bootstrap_samples[index]:
                if train_elem == i:
                    can_go_in_loop = False
                    break
            if can_go_in_loop:
                model = models[index]
                x = train_elem[:-1]
                actual_y = train_elem[-1]
                predicted_y = predict(model, x)
                if (predicted_y != actual_y):
                    misclassified_count += 1
                num_bags_not_having_train_elem += 1
        if (num_bags_not_having_train_elem > 0):
            oob += (misclassified_count / float(num_bags_not_having_train_elem))
            num_of_elements_predicted += 1
    oob /= float(num_of_elements_predicted)
    return oob
if __name__ == "__main__":
    import sys
    c = sys.argv[1]
    for i in range(2, len(sys.argv)):
        c += ' ' + sys.argv[i]
    Xio = np.genfromtxt(c, dtype=np.float64, delimiter=',', skip_header=1)
    rst={}
    #note here 50 is the best hyperparameter value obtained after tuning code attached
    # at the end as comment..where hyperparameter has bn found on using coarse grain search
    bootstrap_samples = get_bootstrap_samples(list(train_XY), 50)
    models=get_trained_models_using_bagging(train_XY, 50,bootstrap_samples)
    with open('predicted_test_Y_rf.csv', 'w') as g:
        writer = csv.DictWriter(g, fieldnames=['t'])
        for test_X in Xio:
            rst['t'] = predict_using_bagging(models, test_X)
            writer.writerow(rst)

#COMPUTING THE VALUE OF HYPERPARAMETER
#One thing to note here is that there are two hyperparameters involved,
# one is no of training_and_testing dataset we are generating from single dataset
# another is the no of attributes we are considering easch time for split_data_set
# but the second normally comes out to be the square root of the first..(through observations and studies it has been shown)
# so we just need to compute the first..which goes on:
# hyperpara_k=[10,15,20,25,30,35,40,45,50]
# least_error=float('inf')
# best_k=0
# for i in hyperpara_k:
#     bootstrap_samples=get_bootstrap_samples(list(train_XY), i)
#     print(type(bootstrap_samples))
#     new_error=get_out_of_bag_error(get_trained_models_using_bagging(train_XY, i,bootstrap_samples), train_XY, bootstrap_samples)
#     if new_error<least_error:
#         least_error=new_error
#         best_k=i
# 50 was obtained from above;
# hyperpara_k=[46,47,48,49,50,51,52,53,54]
# least_error=float('inf')
# best_k=0
# for i in hyperpara_k:
#     bootstrap_samples=get_bootstrap_samples(list(train_XY), i)
#     print(type(bootstrap_samples))
#     new_error=get_out_of_bag_error(get_trained_models_using_bagging(train_XY, i,bootstrap_samples), train_XY, bootstrap_samples)
#     if new_error<least_error:
#         least_error=new_error
#         best_k=i
