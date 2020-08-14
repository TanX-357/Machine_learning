import math
import numpy as np
import csv
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
def compute_ln_norm_distance(vector1, vector2, n):
    vector_len = len(vector1)
    diff_vector = []

    for i in range(0, vector_len):
        abs_diff = abs(vector1[i] - vector2[i])
        diff_vector.append(abs_diff ** n)
    ln_norm_distance = (sum(diff_vector)) ** (1.0 / n)
    return ln_norm_distance


def find_k_nearest_neighbors(train_X, test_example, k, n_in_ln_norm_distance):
    indices_dist_pairs = []
    index = 0
    for train_elem_x in train_X:
        distance = compute_ln_norm_distance(train_elem_x, test_example, n_in_ln_norm_distance)
        indices_dist_pairs.append([index, distance])
        index += 1
    indices_dist_pairs.sort(key=lambda x: x[1])
    top_k_pairs = indices_dist_pairs[0:k]
    top_k_indices = [i[0] for i in top_k_pairs]
    return top_k_indices


def classify_points_using_knn(train_X, train_Y, test_X, n_in_ln_norm_distance, k):
    test_Y = []
    for test_elem_x in test_X:
        top_k_nn_indices = find_k_nearest_neighbors(train_X, test_elem_x, k, n_in_ln_norm_distance)
        top_knn_labels = []
        for i in top_k_nn_indices:
            top_knn_labels.append(train_Y[i])
        most_frequent_label = max(set(top_knn_labels), key=top_knn_labels.count)
        test_Y.append(most_frequent_label)
    return test_Y


def calculate_accuracy(predicted_Y, actual_Y):
    total_num_of_observations = len(predicted_Y)
    num_of_values_matched = 0
    for i in range(0, total_num_of_observations):
        if (predicted_Y[i] == actual_Y[i]):
            num_of_values_matched += 1
    return float(num_of_values_matched) / total_num_of_observations


def get_best_k_using_validation_set(train_X, train_Y, validation_split_percent, n_in_ln_norm_distance):
    mm = -1
    c = 0
    kk = math.floor((100 - validation_split_percent) * len(train_Y) / 100)
    for i in range(2, kk + 1):
        predicted_Y = classify_points_using_knn(train_X[:kk], train_Y[:kk], train_X[kk:], n_in_ln_norm_distance, i)
        r = calculate_accuracy(predicted_Y, train_Y[kk:])
        if (r > mm):
            mm = r
            c = i

    # sorted(r,key=lambda x:(x[0],x[1]))
    return c

train_X = np.genfromtxt('train_X_knn.csv', dtype=np.float64, delimiter=',', skip_header=1)
train_Y = np.genfromtxt('train_Y_knn.csv', dtype=np.float64, delimiter=',', skip_header=0)
# train_X, X_test,train_Y, y_test = train_test_split(X, Y, test_size=0.1)
sc = StandardScaler()
train_X = sc.fit_transform(train_X)
validation_split_percent = 74
n_ln_norm = 3
best_k = get_best_k_using_validation_set(train_X, train_Y, validation_split_percent, n_ln_norm)
k=(best_k)
W=[n_ln_norm,k]
rst={}
with open('WEIGHTS_FILE.csv','w') as WW:
    writer = csv.DictWriter(WW, fieldnames=['t'])
    for i in range(len(W)):
        rst['t'] = W[i]
        writer.writerow(rst)
if __name__ == "__main__":
    import sys

    c = sys.argv[1]
    for i in range(2, len(sys.argv)):
        c += ' ' + sys.argv[i]
    train_data = np.genfromtxt(c, dtype=np.float64, delimiter=',', skip_header=1)
    train_data=sc.fit_transform(train_data)
    train_data = classify_points_using_knn(train_X, train_Y, train_data, n_ln_norm, k)
    rst = {}
    with open('predicted_test_Y_knn.csv', 'w') as g:
        writer = csv.DictWriter(g, fieldnames=['t'])
        for i in range(len(train_data)):
            rst['t'] = int(train_data[i])
            writer.writerow(rst)
