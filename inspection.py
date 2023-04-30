import pandas as pd
import sys
import math

'''
input: 文件路径
output: 数据的Label的list, Label的dict
'''
def Data_Preprocessor(filepath):
    # 因为读入的是tsv，为制表符分隔
    InputData = pd.read_csv(filepath, delimiter='\t')
    Label = []
    LDict = {}  # 创建dict记录每个label出现的次数
    for _, row in InputData.iterrows():
        row_label = row[-1] # 最后一列为label
        Label.append(row_label)
        LDict[row_label] = LDict.get(row_label, 0) + 1
    return Label, LDict

'''
input: Label的dict, 数据样本数(即Label数)
ouput: Label Entropy
'''
def Calc_Label_Entropy(LDict, SampleNum):
    LabelEnt = 0
    for _, value in LDict.items():
        frac = value / SampleNum
        LabelEnt -= frac * math.log(frac, 2)
    return LabelEnt

'''
input: Label的dict, 数据样本数(即Label数)
ouput: Error Rate
'''
def Calc_Error_Rate(LDict, SampleNum):
    maxv = 0
    for _, value in LDict.items():
        maxv = max(maxv, value)
    ErrorRate = (SampleNum - maxv) / SampleNum
    return ErrorRate

if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) != 3:
        print("Usage: python3 inspection.py <input> <output>")
        sys.exit(1)
    
    InputFilePath = sys.argv[1]
    OutputFilePath = sys.argv[2]
    
    Label, LDict = Data_Preprocessor(InputFilePath)
    SampleNum = len(Label)  # 得到总的样本数
    LabelEnt = Calc_Label_Entropy(LDict, SampleNum)
    ErrorRate = Calc_Error_Rate(LDict, SampleNum)

    with open(OutputFilePath, "w") as f:
        f.write(f"entropy: {LabelEnt}\nerror: {ErrorRate}\n")
        f.close()