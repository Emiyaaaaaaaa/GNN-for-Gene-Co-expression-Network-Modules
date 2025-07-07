import numpy as np

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)  # 将预测结果转换为和labels一致的类型
    correct = preds.eq(labels).double()  # T/F-> double() ->0/1
    correct = correct.sum()
    return correct / len(labels)

