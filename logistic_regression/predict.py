import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


X = np.genfromtxt('train_X_lg_v2.csv', dtype=np.float64, delimiter=',', skip_header=1)
y = np.genfromtxt('train_Y_lg_v2.csv', dtype=np.float64, delimiter=',', skip_header=0)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
sc = StandardScaler()

train_X = sc.fit_transform(X_train)
clf = LogisticRegression(penalty='l2',random_state=0).fit(train_X, y_train)

import pickle
saved_model = pickle.dumps(clf)
knn_from_pickle = pickle.loads(saved_model)
rst={}
if __name__ == "__main__":
    import sys
    c = sys.argv[1]
    for i in range(2, len(sys.argv)):
        c += ' ' + sys.argv[i]
    X_test = np.genfromtxt(c, dtype=np.float64, delimiter=',', skip_header=1)
    test_X = sc.fit_transform(X_test)
    train_data = knn_from_pickle.predict(test_X)
    with open('predicted_test_Y_lg.csv', 'w') as g:
        writer = csv.DictWriter(g, fieldnames=['t'])
        for i in range(len(train_data)):
            rst['t'] = train_data[i]
            writer.writerow(rst)