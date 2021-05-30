from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve,auc
import random
import matplotlib.pyplot as plt
import reader

def split_v1(d):
    d1 = []
    d2 = []
    for i in range(len(d)):
        d1.append(d[i][0 : 200])
    for i in range(len(d)):
        d2.append(d[i][200 : len(d[0])])
    return d1, d2

def split_dc(d):
    d1 = []
    d2 = []
    for i in range(len(d)):
        d1.append(d[i][0 : 3044])
    for i in range(len(d)):
        d2.append(d[i][3044 : len(d[0])])
    return d1, d2

def combine(d1, d2):
    d = d1
    for i in range(len(d2)):
        for j in range(len(d2[0])):
            d[i].append(d2[i][j])
    return d

def calc_precision(y_true, y_pred, target):
    tp = 0
    fp = 0
    for i in range(len(y_pred)):
        if target == y_pred[i]:
            if target == y_true[i]:
                tp += 1
            else:
                fp += 1
    if tp + fp == 0:
        return 0
    return tp / (tp + fp)

def calc_recall(y_true, y_pred, target):
    tp = 0
    fn = 0
    for i in range(len(y_true)):
        if target == y_true[i]:
            if target == y_pred[i]:
                tp += 1
            else:
                fn += 1
    if tp + fn == 0:
        return 0
    return tp / (tp + fn)

def calc_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')

def calc_auc(y_true, y_pred, show_roc=False):
    FPR, TPR, threshold = roc_curve(y_true, y_pred, pos_label=1)
    AUC = auc(FPR,TPR)

    if show_roc:
        plt.figure()
        plt.title('ROC CURVE (AUC={:.2f})'.format(AUC))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.0])
        plt.plot(FPR,TPR,color='g')
        plt.plot([0, 1], [0, 1], color='m', linestyle='--')
        plt.show()

    return AUC

def calc_accuracy(y_true, y_pred, normalize=True, sample_weight=None):
    return accuracy_score(y_true, y_pred, normalize, sample_weight)

# 控制0，1，2比例
# d: 特征
# f: 标签
# n0: 0类别/1类别
# n2: 2类别/1类别
def rescale_dc_data(d, f, n0, n2):
    d, f = reader.shuffle(d, f)
    fd = []
    ff = []
    count_0 = n0 * 129
    count_2 = n2 * 129
    for i in range(len(d)):
        if f[i] == 1:
            fd.append(d[i])
            ff.append(f[i])
        else:
            if count_0 > 0 and f[i] == 0:
                count_0 -= 1
                fd.append(d[i])
                ff.append(f[i])
            if count_2 > 0 and f[i] == 2:
                count_2 -= 1
                fd.append(d[i])
                ff.append(f[i])
    return fd, ff
def show_result(y_true, y_pred):
    accuracy = calc_accuracy(y_true, y_pred)
    precision_0 = calc_precision(y_true, y_pred, 0)
    precision_1 = calc_precision(y_true, y_pred, 1)
    recall_0 = calc_recall(y_true, y_pred, 0)
    recall_1 = calc_recall(y_true, y_pred, 1)
    p_0_r_0 = precision_0 * recall_0

    print("accuracy:", accuracy)
    print("precision_0:", precision_0)
    print("precision_1:", precision_1)
    print("recall_0", recall_0)
    print("recall_1:", recall_1)
    print("precision_0 * recall_0:", p_0_r_0)

    return accuracy, precision_0, precision_1, recall_0, recall_1, p_0_r_0

def show_dc_result(y_true, y_pred):
    accuracy = calc_accuracy(y_true, y_pred)
    precision_0 = calc_precision(y_true, y_pred, 0)
    precision_1 = calc_precision(y_true, y_pred, 1)
    precision_2 = calc_precision(y_true, y_pred, 2)
    recall_0 = calc_recall(y_true, y_pred, 0)
    recall_1 = calc_recall(y_true, y_pred, 1)
    recall_2 = calc_recall(y_true, y_pred, 2)
    p_1_r_1 = precision_1 * recall_1
    #f1_score = calc_f1(y_true, y_pred)
    #auc = calc_auc(y_true, y_pred, True)

    print("accuracy:", accuracy)
    print("precision_0:", precision_0)
    print("precision_1:", precision_1)
    print("precision_2:", precision_2)
    print("recall_0", recall_0)
    print("recall_1:", recall_1)
    print("recall_2:", recall_2)
    print("precision_1 * recall_1:", p_1_r_1)
    #print("f1_score:", f1_score)
    #print("auc:", auc)

    return accuracy, precision_0, precision_1, recall_0, recall_1, p_1_r_1

if __name__ == '__main__':
    r = calc_recall([0, 0, 0, 0], [1, 1, 1, 0], 0)
    print(r)