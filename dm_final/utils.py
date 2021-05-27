from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve,auc

import matplotlib.pyplot as plt

def split_v1(d):
    d1 = []
    d2 = []
    for i in range(len(d)):
        d1.append(d[i][0 : 200])
    for i in range(len(d)):
        d2.append(d[i][200 : len(d[0])])
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

def show_result(y_true, y_pred):
    accuracy = calc_accuracy(y_true, y_pred)
    precision_0 = calc_precision(y_true, y_pred, 0)
    precision_1 = calc_precision(y_true, y_pred, 1)
    recall_0 = calc_recall(y_true, y_pred, 0)
    recall_1 = calc_recall(y_true, y_pred, 1)
    f1_score = calc_f1(y_true, y_pred)
    auc = calc_auc(y_true, y_pred, True)

    print("accuracy:", accuracy)
    print("precision_0:", precision_0)
    print("precision_1:", precision_1)
    print("recall_0", recall_0)
    print("recall_1:", recall_1)
    print("f1_score:", f1_score)
    print("auc:", auc)

    return accuracy, precision_0, precision_1, recall_0, recall_1, f1_score, auc

if __name__ == '__main__':
    r = calc_recall([0, 0, 0, 0], [1, 1, 1, 0], 0)
    print(r)