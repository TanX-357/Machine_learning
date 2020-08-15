import numpy as np
from sklearn.model_selection import KFold
import csv
X = np.genfromtxt('train_X_re.csv', dtype=np.float64, delimiter=',', skip_header=1)
y = np.genfromtxt('train_Y_re.csv', dtype=np.float64, delimiter=',', skip_header=0)

def predict(root, X):
    node = root
    while node.left:
        if X[node.feature_index] < node.threshold:
            node = node.left
        else:
            node = node.right
    return node.predicted_class
class Node:
    def __init__(self, predicted_class, depth,mse,threshold=0,parent=None,leaf=0,feature_index=0):
        self.predicted_class = predicted_class
        self.feature_index = feature_index
        self.threshold = threshold
        self.depth = depth
        self.mse=mse
        self.parent=parent
        self.leaf=leaf
        self.left = None
        self.right = None
def create_copy_of_node(nnode):
    given_node=Node(nnode.predicted_class,nnode.depth,nnode.mse,nnode.threshold,nnode.parent,nnode.leaf,nnode.feature_index)
    if nnode.left is not None:
        given_node.left=create_copy_of_node(nnode.left)
    if nnode.right is not None:
        given_node.right=create_copy_of_node(nnode.right)
    return given_node

def mse_for_given_tree(node):
    if node.leaf==1:
        return node.mse
    else:
        if node.left is not None and node.right is not None:
            return mse_for_given_tree(node.left)+mse_for_given_tree(node.right)


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


def get_best_thresold_to_split(X, Y, thresholds,j):
    m = len(X)
    errors =[]
    for threshold in thresholds:
        left = []
        right = []
        for i in range(0,m):
            if X[i][j] < threshold:
                left.append(i)
            else:
                right.append(i)
        left_train_Y = [Y[i] for i in left]
        right_train_Y = [Y[i] for i in right]

        left_avg = sum(left_train_Y)/len(left)
        right_avg = sum(right_train_Y)/len(right)

        left_predicted = np.array([left_avg]*len(left))
        difference =  left_predicted - np.array(left_train_Y )
        left_error = np.sum(np.square(difference))

        right_predicted = np.array([right_avg]*len(right))
        difference =  right_predicted - np.array(right_train_Y )
        right_error = np.sum(np.square(difference))

        error = left_error + right_error
        errors.append(error)
    min_error = min(errors)
    i = errors.index(min_error)
    better_threshold = thresholds[i]

    return better_threshold,min_error
def get_best_split(X,Y):
    best_threshold=None
    best_feature=None
    best_mse=float('inf')
    no_of_features=len(X[0])
    for i in range(no_of_features):
        list_of_values_of_particular_feature=list(set(X[:,i]))
        if len(list_of_values_of_particular_feature)>1:
            possible_thresholds=[]
            for ii in range(1,len(list_of_values_of_particular_feature)):
                possible_thresholds.append((list_of_values_of_particular_feature[ii]+list_of_values_of_particular_feature[ii-1])/2)
            local_threshold,local_mse=get_best_thresold_to_split(X,Y,possible_thresholds,i)
            if local_threshold is not None:
                if best_mse>local_mse:
                    best_mse=local_mse
                    best_threshold=local_threshold
                    best_feature=i
    return best_feature,best_threshold

def construct_tree(X, Y, min_size, depth):
#     classes = list(set(Y))
    predicted_class = np.sum(Y)/len(Y)
    predicted = np.array([predicted_class]*len(Y))
    difference =  predicted - np.array(Y)
    mse = np.sum(np.square(difference))
    node = Node(predicted_class, depth,mse)
    #check is pure
    if len(set(Y)) == 1:
        node.leaf=1
        return node

    #check min subset at node
    if len(Y) <= min_size:
        node.leaf=1
        return node

    feature_index, threshold = get_best_split(X, Y)


    if feature_index is None or threshold is None:
        node.leaf=1
        return node

    node.feature_index = feature_index
    node.threshold = threshold
    left_X, left_Y, right_X, right_Y = split_data_set(X, Y, feature_index, threshold)

    node.left = construct_tree(np.array(left_X), np.array(left_Y), min_size, depth + 1)
    node.right = construct_tree(np.array(right_X), np.array(right_Y), min_size, depth + 1)
    node.left.parent=node
    node.right.parent=node
    return node
def dict_of_pair_of_nodes_with_atleast_one_of_the_two_is_leaf(n):
    d={}
    if n.leaf==1:
        if n.parent.right is not n:
            d[n]=n.parent.right
        else:
            d[n.parent.left]=n
    else:
        d=dict_of_pair_of_nodes_with_atleast_one_of_the_two_is_leaf(n.left)
        d.update(dict_of_pair_of_nodes_with_atleast_one_of_the_two_is_leaf(n.right))
    return d


def tree_size(root):
    sizee = 0
    if root is not None:
        if root.left is not None:
            sizee += tree_size(root.left)
        if root.right is not None:
            sizee += tree_size(root.right)
        sizee += 1
    return sizee


def remove_subtree(node):
    if node.left is not None:
        remove_subtree(node.left)
    if node.right is not None:
        remove_subtree(node.right)
    node = None
    return


def reduce_nodes(NODES):
    global curr_tree_size
    d = dict_of_pair_of_nodes_with_atleast_one_of_the_two_is_leaf(NODES)
    if (len(d)):
        #         l is actually list of pair_of_nodes_with_atleast_one_of_the_two_is_leaf
        l = []
        for k, v in d.items():
            l.append([k, v, tree_size(k) + tree_size(v)])
        l.sort(
            key=lambda k: (-1 * mse_for_given_tree(k[0]) - 1 * mse_for_given_tree(k[1]) + k[0].parent.mse) / k[
                2])
        old_tree_mse = 0
        for i in range(len(l)):
            if l[i][0].leaf == 1:
                old_tree_mse += l[i][0].mse
            if l[i][1].leaf == 1:
                old_tree_mse += l[i][1].mse
        change_in_mse = -1 * mse_for_given_tree(l[0][0]) - 1 * mse_for_given_tree(l[0][1]) + l[0][0].parent.mse
        curr_tree_size = curr_tree_size - tree_size(l[0][0]) - tree_size(l[0][1])
        l[0][0].parent.leaf = 1
        l[0][0].parent.left = None
        l[0][0].parent.right = None
        remove_subtree(l[0][0])
        remove_subtree(l[0][1])
        return change_in_mse + old_tree_mse, curr_tree_size


nnode=construct_tree(X,y,20,0)
tree_list=[]
curr_tree_size=tree_size(nnode)
tree_list.append([create_copy_of_node(nnode),mse_for_given_tree(nnode),curr_tree_size])
while nnode.left is not None and nnode.right is not None:
    new_mse,new_size=reduce_nodes(nnode)
    new_node=create_copy_of_node(nnode)
    tree_list.append([new_node,new_mse,new_size])
tree_list[0].append(0)
x=[]
x.append(0)
for i in range(1,len(tree_list)):
    tree_list[i].append((tree_list[i][1]-tree_list[i-1][1])/(tree_list[i-1][2]-tree_list[i][2]))
    x.append((tree_list[i][1]-tree_list[i-1][1])/(tree_list[i-1][2]-tree_list[i][2]))

trr = [] #HOLDING TREE SCORE FOR DIFFERENT VALUES OF OUR HYPERPARAME
for i in range(len(x)):
    trr.append(0)
from sklearn.model_selection import KFold

kf = KFold(n_splits=10)
kf.get_n_splits(X)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    root_node = construct_tree(X_train, y_train, 20, 0)
    curr_tree_size = tree_size(root_node)
    mse_plus_tr_sz = [] # USED FOR HOLDING MSE FOR A GIVEN PRUNED TREE AND THE PRUNED TREE-SIZE
    xr = []
    yy = []

    indices = 1
    for im in X_test:
        yy.append(predict(root_node, im)) #predicting he output labels for a  full tree(no_pruning)
    right_predicted = np.array(yy)
    difference = right_predicted - np.array(y_test)
    trr[0] = trr[0] + np.sum(np.square(difference))
    mse_plus_tr_sz.append([mse_for_given_tree(root_node), curr_tree_size])
    st = [] #using it just to store the previous value of ss
    copy_of_root_node = create_copy_of_node(root_node)
    ss = list(reduce_nodes(root_node))
    while root_node.right is not None and root_node.left is not None and indices < len(x):
        yy = []
        # pruning the tree based on given x[i] value
        while root_node.right is not None and root_node.left is not None and (
                (ss[0] - mse_plus_tr_sz[-1][0]) / (mse_plus_tr_sz[-1][1] - ss[1]) <= x[indices]):
            st = ss
            copy_of_root_node = create_copy_of_node(root_node)
            ss = list(reduce_nodes(root_node))
        if root_node.right is not None:
            if (len(st)):
                mse_plus_tr_sz.append(st)
            else:
                mse_plus_tr_sz.append(ss)
            for im in X_test:
                yy.append(predict(copy_of_root_node, im)) #predicting ouput for a given pruned tree 
            right_predicted = np.array(yy)
            difference = right_predicted - np.array(y_test)
            trr[indices] = trr[indices] + np.sum(np.square(difference)) #adding mse value for a given pruned tree for a given test and train data for given x[i] value(hyperparameter)
            indices += 1
        else:
            while len(mse_plus_tr_sz) < len(x):
                yy = []
                if ((ss[0] - mse_plus_tr_sz[-1][0]) / (mse_plus_tr_sz[-1][1] - ss[1]) <= x[indices]):
                    mse_plus_tr_sz.append(ss)
                    for im in X_test:
                        yy.append(predict(root_node, im))
                    right_predicted = np.array(yy)
                    difference = right_predicted - np.array(y_test)
                    trr[indices] = trr[indices] + np.sum(np.square(difference))
                    indices += 1
                else:
                    mse_plus_tr_sz.append(st)
                    for im in X_test:
                        yy.append(predict(copy_of_root_node, im))
                    right_predicted = np.array(yy)
                    difference = right_predicted - np.array(y_test)
                    trr[indices] = trr[indices] + np.sum(np.square(difference))
                    indices += 1

best_val_of_hyper = 0
for i in range(1, len(trr)):
    if trr[i] < trr[best_val_of_hyper]:
        best_val_of_hyper= i



if __name__ == "__main__":
    import sys
    c = sys.argv[1]
    for i in range(2, len(sys.argv)):
        c += ' ' + sys.argv[i]
    test_data = np.genfromtxt(c, dtype=np.float64, delimiter=',', skip_header=1)
    for i in range(len(X)):
        yy = []
        for im in test_data:
            yy.append(predict(tree_list[best_val_of_hyper][0], im))
    rst={}
    with open('predicted_test_Y_re.csv', 'w') as g:
        writer = csv.DictWriter(g, fieldnames=['t'])
        for i in range(len(yy)):
            rst['t'] = yy[i]
            writer.writerow(rst)

