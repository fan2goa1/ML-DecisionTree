import sys
import pandas as pd
import math
import time

NodeCnt = 0

class TreeNode:
    def __init__(self, NodeID):
        self.id = NodeID     # 节点编号
        self.isLeaf = 0 # 1表示此节点为叶节点, 0不是
        self.crit = 0   # 此节点用于划分的属性序号
        self.identify = None # 分得的label
        self.child = {} # 分类后的子节点编号
        return 
    
    def SetLeaf(self, X, label_name):  # 将当前节点设为叶节点, 并分类
        self.isLeaf = 1
        cnt0 = sum(1 for sample in X if sample[-1] == label_name[0])
        cnt1 = sum(1 for sample in X if sample[-1] == label_name[1])
        if cnt0 > cnt1:
            self.identify = label_name[0]
        elif cnt0 < cnt1:
            self.identify = label_name[1]
        else :
            if label_name[0] > label_name[1]:
                self.identify = label_name[0]
            else :
                self.identify = label_name[1]
        return
    
    def AddNode(self, Type, ChildNode):
        self.child[Type] = ChildNode
        return 

# 计算指定属性的互信息
def Calc_Mutual_Information(V0, V1, label_name):
    TotEnt, Ent0, Ent1 = 0, 0, 0
    # 分别统计两个子数据集的两个标签的样本数
    cnt00 = sum(1 for sample in V0 if sample[-1] == label_name[0])
    cnt01 = sum(1 for sample in V0 if sample[-1] == label_name[1])
    cnt10 = sum(1 for sample in V1 if sample[-1] == label_name[0])
    cnt11 = sum(1 for sample in V1 if sample[-1] == label_name[1])
    num0, num1 = len(V0), len(V1) # 第一、二个子数据集的样本数
    cnt0 = cnt00 + cnt10 # 总的为label[0]的样本数
    cnt1 = cnt01 + cnt11
    frac0 = cnt0 / (cnt0+cnt1)
    frac1 = cnt1 / (cnt0+cnt1)
    if cnt0 != 0 and cnt1 != 0:
        TotEnt = -(frac0)*math.log(frac0, 2) - frac1*math.log(frac1, 2)
    frac0 = cnt00 / num0
    frac1 = cnt01 / num0
    if cnt00 != 0 and cnt01 != 0:
        Ent0 = -(frac0)*math.log(frac0, 2) - frac1*math.log(frac1, 2)
    frac0 = cnt10 / num1
    frac1 = cnt11 / num1
    if cnt10 != 0 and cnt11 != 0:
        Ent1 = -(frac0)*math.log(frac0, 2) - frac1*math.log(frac1, 2)
    p0 = num0 / (num0+num1)
    p1 = num1 / (num0+num1)
    MI = TotEnt - p0*Ent0 - p1*Ent1
    return MI

# 计算指定属性的增益率
def Calc_Gain_Rate(V0, V1, label_name):
    IG = Calc_Mutual_Information(V0, V1, label_name)
    num0, num1 = len(V0), len(V1)
    frac0 = num0 / (num0+num1)
    frac1 = num1 / (num0+num1)
    IV = - frac0*math.log(frac0, 2) - frac1*math.log(frac1, 2)
    Gain_Rate = IG / IV
    return Gain_Rate

# 计算指定属性的Gini系数
def Calc_Gini(V0, V1, label_name):
    num0, num1 = len(V0), len(V1)
    frac0 = num0 / (num0+num1)
    frac1 = num1 / (num0+num1)
    cnt00 = sum(1 for sample in V0 if sample[-1] == label_name[0])
    cnt01 = sum(1 for sample in V0 if sample[-1] == label_name[1])
    cnt10 = sum(1 for sample in V1 if sample[-1] == label_name[0])
    cnt11 = sum(1 for sample in V1 if sample[-1] == label_name[1])
    # 计算两个子数据集的Gini系数
    Gini0 = 1 - (cnt00/num0)**2 - (cnt01/num0)**2
    Gini1 = 1 - (cnt10/num1)**2 - (cnt11/num1)**2
    Gini = frac0*Gini0 + frac1*Gini1
    return Gini

# 划分当前数据集的递归函数
def split(X, nowNode, now_depth, max_depth, isSplited, attr_name, label_name):
    if now_depth == max_depth:   # 到达最大深度
        nowNode.SetLeaf(X, label_name)  # 标记叶节点并分类
        return
    
    # 如果当前待分数据集的label都一样则标为叶节点
    if all(sample[-1] ==  X[0][-1] for sample in X):
        nowNode.SetLeaf(X, label_name)
        return 
    # 选择最合适的attribute
    max_ent_ind, max_MI = -1, 0
    for i, tag in enumerate(isSplited):
        if tag == 1:    # 该属性已被用于过划分, 跳过
            continue
        # 获取某一特定属性值的子数据集
        V0 = [sample for sample in X if sample[i] == attr_name[i][0]]
        V1 = [sample for sample in X if sample[i] == attr_name[i][1]]
        if len(V0) == 0 or len(V1) == 0:
            continue
        MI = Calc_Mutual_Information(V0, V1, label_name) # 互信息
        # MI = Calc_Gain_Rate(V0, V1, label_name)   # 增益率
        # MI = Calc_Gini(V0, V1, label_name)        # Gini系数
        # print(attr_title[i], MI)
        if MI > max_MI: # 更新当前最大MI的属性
            max_MI = MI
            max_ent_ind = i

    # 没有可选的attribute
    if max_ent_ind == -1:
        nowNode.SetLeaf(X, label_name)  # 标记叶节点并分类
        return 
    
    # 划分并递归
    global NodeCnt
    nowNode.crit = max_ent_ind
    isSplited[max_ent_ind] = 1
    now_depth += 1
    NodeCnt += 1
    newNode = TreeNode(NodeCnt)
    nowNode.AddNode(attr_name[max_ent_ind][0], newNode)
    V0 = [sample for sample in X if sample[max_ent_ind] == attr_name[max_ent_ind][0]]
    split(V0, newNode, now_depth, max_depth, isSplited, attr_name, label_name)
    
    NodeCnt += 1
    newNode = TreeNode(NodeCnt)
    nowNode.AddNode(attr_name[max_ent_ind][1], newNode)
    V1 = [sample for sample in X if sample[max_ent_ind] == attr_name[max_ent_ind][1]]
    split(V1, newNode, now_depth, max_depth, isSplited, attr_name, label_name)
    now_depth -= 1  # 回溯
    isSplited[max_ent_ind] = 0  # 回溯
    return 

# 生成决策树的总函数  
def Gen_Tree(X, max_depth, attr_name, label_name, attr_title):
    global NodeCnt
    NodeCnt += 1
    root = TreeNode(NodeCnt)
    isSplited = [0 for i in range(len(attr_name))]
    split(X, root, 0, max_depth, isSplited, attr_name, label_name)
    return root

# 读入并处理数据
def Read_Data(trainfile, testfile):
    InputData_train = pd.read_csv(trainfile, delimiter='\t')
    InputData_test = pd.read_csv(testfile, delimiter='\t')
    # 将训练、测试数据集转成字符串的list
    train_input = InputData_train.values.tolist()
    train_input = [[str(elem) for elem in data] for data in train_input]
    test_input = InputData_test.values.tolist()
    test_input = [[str(elem) for elem in data] for data in test_input]
    num_rows, num_cols = InputData_train.shape
    attr_name = []  # 每个属性的两个取值
    label_name = [] # label的名字
    attr_title = [] # 属性的名称
    
    for _, col in InputData_train.iloc[:, :-1].items():
        col_unique = col.unique().tolist()
        # 此属性只有一个取值,为方便后续计算,添加一个空取值
        if len(col_unique) < 2: 
            col_unique.append("null")
        col_unique = [str(elem) for elem in col_unique] # 转为字符串
        col_unique.sort()
        attr_name.append(col_unique)
    
    label_name = InputData_train.iloc[:, -1].unique().tolist()
    label_name = [str(elem) for elem in label_name]
    label_name.sort()
    
    attr_title = InputData_train.columns.to_list()[:-1]
    attr_title = [str(elem) for elem in attr_title]

    return train_input, test_input, attr_name, label_name, attr_title

# 对给定数据集进行预测并输出
def Prediction(root, X, out_file):
    # 预测
    pred_label = []
    for data in X:
        nowNode = root
        while nowNode.isLeaf == 0:
            attr = data[nowNode.crit]
            nowNode = nowNode.child[attr]
        label = nowNode.identify
        pred_label.append(label)
    # 输出到文件
    with open(out_file, "w") as f:
        for x in pred_label:
            f.write(str(x) + "\n")
    # 计算并返回错误率
    err_cnt = 0
    for i in range(len(pred_label)):
        if pred_label[i] != X[i][-1]:
            err_cnt += 1
    return err_cnt / len(pred_label)

# 输出决策树的具体信息
def printTree(nowNode, X, dep, attr_name, label_name, attr_title):
    if dep == 0: # root节点
        cnt0, cnt1 = 0, 0
        for sample in X:
            if sample[-1] == label_name[0]:
                cnt0 += 1
            else:
                cnt1 += 1
        print("["+str(cnt0)+" "+label_name[0]+"/"+str(cnt1)+" "+label_name[1]+"]")
    
    if nowNode.isLeaf == 1: # 如果为叶节点则返回
        return

    feature_id = nowNode.crit
    V0 = [sample for sample in X if sample[feature_id] == attr_name[feature_id][0]]
    V1 = [sample for sample in X if sample[feature_id] == attr_name[feature_id][1]]
    # 统计两个label的样本数量
    cnt0, cnt1 = 0, 0
    for sample in V0:
        if sample[-1] == label_name[0]:
            cnt0 += 1
        else:
            cnt1 += 1
    HeaderMsg = "| "*(dep+1)+attr_title[feature_id]+" = "
    print(HeaderMsg+attr_name[feature_id][0]+": "
          +"["+str(cnt0)+" "+label_name[0]+"/"+str(cnt1)+" "+label_name[1]+"]")
    printTree(nowNode.child[attr_name[feature_id][0]], V0, dep+1, attr_name, label_name, attr_title)
    cnt0, cnt1 = 0, 0
    for sample in V1:
        if sample[-1] == label_name[0]:
            cnt0 += 1
        else:
            cnt1 += 1
    print(HeaderMsg+attr_name[feature_id][1]+": "
          +"["+str(cnt0)+" "+label_name[0]+"/"+str(cnt1)+" "+label_name[1]+"]")
    printTree(nowNode.child[attr_name[feature_id][1]], V1, dep+1, attr_name, label_name, attr_title)
    return 

if __name__ == "__main__":
    if len(sys.argv) < 7:
        print("Usage: python3 decisionTree.py <train input> <test input> <max depth> <train out> <test out> <metrics out>")
        sys.exit(0)
    
    max_depth = int(sys.argv[3])
    train_out, test_out = sys.argv[4], sys.argv[5]
    metrics_out = sys.argv[6]
    
    # 从输入文件得到训练集、测试集的list, 每个属性的两个值
    train_input, test_input, attr_name, label_name, attr_title = Read_Data(sys.argv[1], sys.argv[2])
    # 生成决策树
    Tree = Gen_Tree(train_input, max_depth, attr_name, label_name, attr_title)
    # 根据生成的决策树预测训练集并输出到文件
    error_train = Prediction(Tree, train_input, train_out)
    # 根据生成的决策树预测测试集并输出到文件
    error_test = Prediction(Tree, test_input, test_out)
    # 输出error rate到metrics_out
    with open(metrics_out, "w") as f:
        f.write("error(train): " + str(error_train) + "\n")
        f.write("error(test): " + str(error_test))
    # 输出决策树
    printTree(Tree, train_input, 0, attr_name, label_name, attr_title)