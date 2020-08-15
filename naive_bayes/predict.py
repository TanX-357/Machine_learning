def openfile(filename):
    with open(filename,'r') as f:
        data=f.readlines()
    return data
import numpy as np
import csv

train_Y = np.genfromtxt('train_Y_nb.csv', dtype=np.float64, delimiter=',', skip_header=0)


def remove_spl_chars_except_space(s):
    i = 0
    s_with_no_spl_chars = ""
    # using ASCII Values of characters
    for i in range(len(s)):
        if (ord(s[i]) >= ord('A') and
                ord(s[i]) <= ord('Z') or
                ord(s[i]) >= ord('a') and
                ord(s[i]) <= ord('z') or
                ord(s[i]) == ord(' ')):
            s_with_no_spl_chars += s[i]

    return s_with_no_spl_chars


def preprocessing(s):
    s = remove_spl_chars_except_space(s)
    s = ' '.join(s.split())  # replaces multiple spaces with single space
    s = s.lower()  # convert to lowercase
    return s

train_data=openfile('train_X_nb.csv')
X_train=[]
for i in train_data:
    X_train.append(preprocessing(i))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train,train_Y, test_size=0.40, random_state=0)
def class_wise_words_frequency_dict(X, Y):
    class_wise_frequency_dict = dict()
    for i in range(len(X)):
        words = X[i].split()
        for token_word in words:
            y = Y[i]
            if y not in class_wise_frequency_dict:
                class_wise_frequency_dict[y] = dict()
            if token_word not in class_wise_frequency_dict[y]:
                class_wise_frequency_dict[y][token_word] = 0
            class_wise_frequency_dict[y][token_word] += 1
    return class_wise_frequency_dict


def compute_prior_probabilities(Y):
    classes = list(set(Y))
    n_docs = len(Y)
    prior_probabilities = dict()
    for c in classes:
        prior_probabilities[c] = Y.count(c) / n_docs
    return prior_probabilities


def get_class_wise_denominators_likelihood(X, Y, class_wise_frequency_dict):
    #     class_wise_frequency_dict = class_wise_words_frequency_dict(X, Y)
    classes = list(set(Y))
    class_wise_denominators = dict()
    vocabulary = []
    for c in classes:
        frequency_dict = class_wise_frequency_dict[c]
        class_wise_denominators[c] = sum(list(frequency_dict.values()))
        vocabulary += list(frequency_dict.keys())

    vocabulary = list(set(vocabulary))

    return class_wise_denominators, len(vocabulary)


def compute_likelihood(test_X, c, class_wise_frequency_dict, class_wise_denominators, alpha, vocab,
                       prior_probabilities):
    likelihood = 0
    words = test_X.split()

    for word in words:
        count = 0
        words_frequency = class_wise_frequency_dict[c]
        if word in words_frequency:
            count = class_wise_frequency_dict[c][word]
        likelihood += np.log((count + alpha) / (class_wise_denominators[c] + vocab * alpha))
    return likelihood


def predict(test_X, class_wise_frequency_dict, class_wise_denominators, alpha, vocab, prior_probabilities):
    test_y = []
    classes = []
    for u in prior_probabilities.keys():
        classes.append(u)
    for i in test_X:
        best_p = -99999
        best_c = -1
        for c in classes:
            p = compute_likelihood(i, c, class_wise_frequency_dict, class_wise_denominators, alpha, vocab,
                                   prior_probabilities) + np.log(prior_probabilities[c])
            if p > best_p:
                best_p = p
                best_c = c
        test_y.append(best_c)
    return test_y

cwfd=class_wise_words_frequency_dict(X_train,y_train)
prior_prob=compute_prior_probabilities(list(y_train))
cwdl,vocab=get_class_wise_denominators_likelihood(X_train,y_train,cwfd)
if __name__ == "__main__":
    import sys
    c = sys.argv[1]
    for i in range(2, len(sys.argv)):
        c += ' ' + sys.argv[i]
    t_data=openfile(c)
    X=[]
    for i in t_data:
        X.append(preprocessing(i))
    rst = {}
    final_Y=predict(X,cwfd,cwdl,0.01,vocab,prior_prob) #computed 0.01(smoothening) through hypertuning whose code i have included in the end as comment
    with open('predicted_test_Y_nb.csv', 'w') as g:
        writer = csv.DictWriter(g, fieldnames=['t'])
        for i in range(len(final_Y)):
            rst['t'] = final_Y[i]
            writer.writerow(rst)
# THE CODE WHERE I COMPUTED THE VALUE OF SMOOTHENING CONSTANT THROUGH COARSE GRAIN METHOD
# from sklearn import metrics
# PROBABLE_SMOOTHENING_VALUE=[0.00001,0.0001,0.001,0.01,0.1]
# best_accuracy=0
# best_smoothening_value=0
# for i in PROBABLE_SMOOTHENING_VALUE:
#     Y_actual=predict(X_test,cwfd,cwdl,i,vocab,prior_prob)
#     if(metrics.accuracy_score(y_test, Y_actual)>best_accuracy):
#         best_accuracy=metrics.accuracy_score(y_test,Y_actual)
#         best_smoothening_value=i