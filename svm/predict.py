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
ll=[1,10,100,1000,10000]
hh=[]
for i in range(len(ll)):
    clf = make_pipeline(StandardScaler(), SVC(C=ll[i], gamma='auto'))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    hh.append(metrics.accuracy_score(y_test, y_pred))
qq=0

for i in range(1,len(ll)):
    if(hh[qq]<hh[i]):
        qq=i
if qq is not 0:
    ol=hh[qq]
    a=ll[qq-1]
    b=ll[qq]
    while(abs(a-b)>0.01):
        m=a+(b-a)/2
        clf = make_pipeline(StandardScaler(), SVC(C=m,gamma='auto'))
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if(ol>metrics.accuracy_score(y_test, y_pred)):
            a=m
        else:
            ol=metrics.accuracy_score(y_test, y_pred)
            b=m
else:
    ol=0
if qq is not len(ll)-1:            
    ool=hh[qq]
    aa=ll[qq]
    bb=ll[qq+1]
    while(abs(aa-bb)>0.01):
        m=aa+(bb-aa)/2
        clf = make_pipeline(StandardScaler(), SVC(C=m,gamma='auto'))
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if(ool>metrics.accuracy_score(y_test, y_pred)):
            bb=m
        else:
            ool=metrics.accuracy_score(y_test, y_pred)
            aa=m
else:
    ool=0
if(ol>ool):
    c=b
else:
    c=aa
clf = make_pipeline(StandardScaler(), SVC(C=c,gamma='auto'))
clf.fit(X_train, y_train)
   

import pickle
saved_model = pickle.dumps(clf)
knn_from_pickle = pickle.loads(saved_model)
rst={}
if __name__ =='__main__':
    import sys
    c=sys.argv[1]
    for i in range(2,len(sys.argv)):
        c+=' '+sys.argv[i]
    X_test=np.genfromtxt(c, dtype=np.float64, delimiter=',', skip_header=1)
    train_data=knn_from_pickle.predict(X_test)
    with open('predicted_test_Y.csv', 'w') as g:
        writer = csv.DictWriter(g, fieldnames=['t'])
        for i in range(len(train_data)):
            rst['t'] = train_data[i]
            writer.writerow(rst)
