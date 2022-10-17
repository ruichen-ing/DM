# %%
import math
import queue
import pandas as pd
import copy


# construct a node class for the decision tree
class Node(object):
    # target: [xx, yy] xx: number of tuples having target=0, yy: number of tuples having target=1
    # feature: first row: cp, second row: exang, last row: thal
    # split_feature: criterion feature used to split the node
    # split_value: the value of the spilt_feature used to split
    def __init__(self, features, target, impurity_value, split_feature="None", split_value=-1, isLeaf=True):
        self.features = features
        self.target = target
        self.impurity_value = impurity_value
        self.split_feature = split_feature
        self.split_value = split_value
        self.left = None
        self.right = None
        self.isLeaf = isLeaf

    def print_node(self):
        print("node: ", self)
        print("features: ", self.features)
        print("target[disease, no disease]: ", self.target)
        print("impurity: ", self.impurity_value)
        print("split_feature: ", self.split_feature)
        print("split_value: ", self.split_value)
        print("left node: ", self.left)
        print("right node: ", self.right)
        print("is Leaf: ", self.isLeaf)
        print()


# functions used for decision tree

# calculate impurity based on entropy
def impurity(p1, p2):
    if p1 == 0 or p2 == 0:
        return 0
    return - p1 * math.log(p1, 2) - p2 * math.log(p2, 2)


def cost(left_data, right_data, last_impurity):
    N_left = left_data.shape[0]
    N1_left = left_data[left_data['target'] == 1].shape[0]
    N0_left = N_left - N1_left

    N_right = right_data.shape[0]
    N1_right = right_data[right_data['target'] == 1].shape[0]
    N0_right = N_right - N1_right

    ip_left = impurity(N1_left / N_left, N0_left / N_left)
    ip_right = impurity(N1_right / N_right, N0_right / N_right)

    return ip_left, ip_right, (N_left * ip_left + N_right * ip_right) / (N_left + N_right) - last_impurity


def make_decision(root, data):
    # split(root, data)
    # spilt dataset to find the division with the lowest cost
    # one-vs-All: we always pick one value of the feature and compute the cost. Eventually, we get the mini cost and the corresbonding feature division

    # TODO
    # what if the parent nodes has the lowest cost?

    # mini: [mini cost, feature, feature value, left impurity, right impurity]
    mini = [float('inf'), "", -1, 0, 0]

    # get mini cost
    for i in range(0, len(root.features)):
        if len(root.features[i]) == 1:
            continue
        if i == 0:
            current_feature = 'cp'
        elif i == 1:
            current_feature = 'exang'
        else:
            current_feature = 'thal'
        for j in range(0, len(root.features[i])):
            current_value = root.features[i][j]
            left_data = data[data[current_feature] == current_value]
            right_data = data[data[current_feature] != current_value]
            ip_left, ip_right, cost_res = cost(left_data, right_data, root.impurity_value)
            cal = [cost_res, current_feature, current_value, ip_left, ip_right]
            if cal[0] < mini[0]:
                mini = cal

    # update info in root
    root.split_feature = mini[1]
    root.split_value = mini[2]
    root.isLeaf = False

    # split root into two subtrees, generate left and right node
    left_data = data[data[mini[1]] == mini[2]]
    left_num_false = left_data[left_data['target'] == 0].shape[0]
    left_num_true = left_data[left_data['target'] == 1].shape[0]

    right_data = data[data[mini[1]] != mini[2]]
    right_num_false = right_data[right_data['target'] == 0].shape[0]
    right_num_true = right_data[right_data['target'] == 1].shape[0]

    if mini[1] == 'cp':
        index = 0
        left_features = [[mini[2]], root.features[1], root.features[2]]
    elif mini[1] == 'exang':
        index = 1
        left_features = [root.features[0], [mini[2]], root.features[2]]
    elif mini[1] == 'thal':
        index = 2
        left_features = [root.features[0], root.features[1], [mini[2]]]

    new_feature = copy.deepcopy(root.features)
    new_feature[index].remove(mini[2])
    right_features = new_feature

    left = Node(left_features, [left_num_false, left_num_true], mini[3])
    right = Node(right_features, [right_num_false, right_num_true], mini[4])

    return left, right, left_data, right_data


def build_tree(root, data, level):
    if level == 3:
        return

    left, right, left_data, right_data = make_decision(root, data)
    root.left = left
    root.right = right
    build_tree(left, left_data, level + 1)
    build_tree(right, right_data, level + 1)


def predict(root, data):
    # 取出来的每一行是一个tuple，tuple第一个元素是index，第二个元素是所有数据组成的series数组
    # 因此要取tuple中的某个属性值得数据，需要row[1].属性
    res = 0
    for row in data.iterrows():
        if predict_helper(root, row[1]) == row[1].target:
            res = res + 1
    return res


def predict_helper(root, data):
    if root.isLeaf:
        if root.target[0] > root.target[1]:
            return 0
        else:
            return 1

    split_feature = root.split_feature
    split_value = root.split_value
    # 获取data中feature为split_feature的值，与split_value比较大小，从而得知是往左还是往右
    if split_feature == 'cp':
        data_value = data.cp
    elif split_feature == 'exang':
        data_value = data.exang
    elif split_feature == 'thal':
        data_value = data.thal

    if data_value == split_value:
        return predict_helper(root.left, data)
    else:
        return predict_helper(root.right, data)


def print_tree(root):
    q = queue.Queue()
    q.put(root)
    while not q.empty():
        node = q.get()
        node.print_node()
        if node.left is not None:
            q.put(node.left)
        if node.right is not None:
            q.put(node.right)


if __name__ == "__main__":
    heart_train = pd.read_csv('./data/heart_train_data.csv')
    heart_validate = pd.read_csv('./data/heart_validate_data.csv')

    numFalse = heart_train[heart_train['target'] == 0].shape[0]
    numTrue = heart_train[heart_train['target'] == 1].shape[0]

    N = heart_train.shape[0]
    N1 = heart_train[heart_train['target'] == 1].shape[0]
    N0 = N - N1
    impurity_value = impurity(N1 / N, N0 / N)

    root = Node([[0, 1, 2, 3], [0, 1], [1, 2, 3]],
                [numFalse, numTrue],
                impurity_value,
                "None",
                -1)

    # training
    build_tree(root, heart_train, 0)
    print_tree(root)

    # validate the prediction
    acc = predict(root, heart_validate) / heart_validate.shape[0]
    print("The accuracy of the decision tree tested by validation set: ", acc)
