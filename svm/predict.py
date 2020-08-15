import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
X = np.genfromtxt('train_X_svm.csv', dtype=np.float64, delimiter=',', skip_header=1)
Y = np.genfromtxt('train_Y_svm.csv', dtype=np.float64, delimiter=',', skip_header=0)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
from sklearn.svm import SVC
from sklearn import metrics
prob_hyp_val=[1,10,100,1000,10000]
acc_list_for_diff_hyp_val=[]
for i in range(len(prob_hyp_val)):
    clf = make_pipeline(StandardScaler(), SVC(C=prob_hyp_val[i], gamma='auto'))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc_list_for_diff_hyp_val.append(metrics.accuracy_score(y_test, y_pred))
best_hyp_val_after_first_search=0

for i in range(1,len(prob_hyp_val)):
    if(acc_list_for_diff_hyp_val[best_hyp_val_after_first_search]<acc_list_for_diff_hyp_val[i]):
        best_hyp_val_after_first_search=i
if best_hyp_val_after_first_search is not 0:
    best_hyp_val_left_of_prev_best=acc_list_for_diff_hyp_val[best_hyp_val_after_first_search]
    a=prob_hyp_val[best_hyp_val_after_first_search-1]
    b=prob_hyp_val[best_hyp_val_after_first_search]
    while(abs(a-b)>0.01):
        m=a+(b-a)/2
        clf = make_pipeline(StandardScaler(), SVC(C=m,gamma='auto'))
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if(best_hyp_val_left_of_prev_best>metrics.accuracy_score(y_test, y_pred)):
            a=m
        else:
            best_hyp_val_left_of_prev_best=metrics.accuracy_score(y_test, y_pred)
            b=m
else:
    best_hyp_val_left_of_prev_best=0
if best_hyp_val_after_first_search is not len(prob_hyp_val)-1:
    best_hyp_val_right_of_prev_best=acc_list_for_diff_hyp_val[best_hyp_val_after_first_search]
    aa=prob_hyp_val[best_hyp_val_after_first_search]
    bb=prob_hyp_val[best_hyp_val_after_first_search+1]
    while(abs(aa-bb)>0.01):
        m=aa+(bb-aa)/2
        clf = make_pipeline(StandardScaler(), SVC(C=m,gamma='auto'))
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if(best_hyp_val_right_of_prev_best>metrics.accuracy_score(y_test, y_pred)):
            bb=m
        else:
            best_hyp_val_right_of_prev_best=metrics.accuracy_score(y_test, y_pred)
            aa=m
else:
    best_hyp_val_right_of_prev_best=0
if(best_hyp_val_left_of_prev_best>best_hyp_val_right_of_prev_best):
    best_hyp_val=b
else:
    best_hyp_val=aa
clf = make_pipeline(StandardScaler(), SVC(C=best_hyp_val,gamma='auto'))
clf.fit(X_train, y_train)


import pickle
saved_model = pickle.dumps(clf)
svm_from_pickle = pickle.loads(saved_model)
rst={}
if __name__ =='__main__':
    import sys
    c=sys.argv[1]
    for i in range(2,len(sys.argv)):
        c+=' '+sys.argv[i]
    X_test=np.genfromtxt(c, dtype=np.float64, delimiter=',', skip_header=1)
    prediction=svm_from_pickle.predict(X_test)
    with open('predicted_test_Y.csv', 'w') as g:
        writer = csv.DictWriter(g, fieldnames=['t'])
        for i in range(len(prediction)):
            rst['t'] = prediction[i]
            writer.writerow(rst)
