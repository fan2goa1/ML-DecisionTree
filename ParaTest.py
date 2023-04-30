import pandas as pd
import dT_Gini as dt
import math
import sys
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 ParaTest.py <dataset>")
        sys.exit(0)
    
    dataset = sys.argv[1]
    train_input_file = dataset + "_train.tsv"
    test_input_file = dataset + "_test.tsv"
    train_out = "train.labels"
    test_out = "test.labels"
    metrics_out = "test_metrics.txt"

    list_depth = [i for i in range(11)]
    list_train_err, list_test_err = [], []

    for max_depth in range(11):
        # 从输入文件得到训练集、测试集的list, 每个属性的两个值
        train_input, test_input, attr_name, label_name, attr_title = dt.Read_Data(train_input_file, test_input_file)
        # 生成决策树
        Tree = dt.Gen_Tree(train_input, max_depth, attr_name, label_name, attr_title)
        # 根据生成的决策树预测训练集并输出到文件
        error_train = dt.Prediction(Tree, train_input, train_out)
        # 根据生成的决策树预测测试集并输出到文件
        error_test = dt.Prediction(Tree, test_input, test_out)
        list_train_err.append(error_train)
        list_test_err.append(error_test)
    
    # 利用seaborn创建图标
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制折线图
    sns.lineplot(x=list_depth, y=list_train_err, label='Train Error',
                 color='#10454F', linewidth=3, ax=ax)
    sns.lineplot(x=list_depth, y=list_test_err, label='Test Error',
                 color='#A3AB78', linewidth=3, ax=ax)

    # 修改坐标等
    title = "Error Rate on Dataset("+dataset+")"
    ax.set_title(title, fontsize=25)
    ax.set_xlabel("Max depth", fontsize=20)
    ax.set_ylabel("Error Rate", fontsize=20)
    ax.tick_params(labelsize=15)
    ax.legend(fontsize=25)

    figname = dataset + "_Gini.svg"
    plt.savefig(figname, format='svg')