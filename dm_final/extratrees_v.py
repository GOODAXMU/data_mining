import numpy
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2
import feature_selector
import utils
import reader

# b_threshold：类别型特征的总和阈值
# v_threshold：数值型特征的方差阈值
# p_threshold：类别型特征的卡方阈值
# k_part：数据集的等分数量，训练集合占k - 1份，测试集占1份
# n_part：将标签为1的数据划分为两份，并且选择其中的第n份，当n大于1则选取全部
# do_shuffle：是否将原数据打乱
def v1_filter(b_threshold, v_threshold, p_threshold, k_part, n_part, do_shuffle=False):
    # 加载训练数据
    d, f = reader.readVTrainingData_2('./dm_final/data/V1_ECFP4.csv', k_part, n_part, do_shuffle)
    # 将数值型和类别型数据分开
    d1, d2 = utils.split_v1(d)
    # 转numpy后计算每列的方差以及每列的和
    nd1 = numpy.array(d1)
    nd2 = numpy.array(d2)
    vars = numpy.var(nd1, axis=0).tolist()
    sums = numpy.sum(nd2, axis=0).tolist()
    # 卡方校验计算列与标签之间的关联度
    _, pvals = chi2(d2, f)
    # 删除统计值小于给定阈值的所有列
    sum_count = 0
    pval_count = 0
    var_count = 0
    indexes_of_removed_d1 = []
    indexes_of_removed_d2 = []
    for i in range(len(vars)):
        if vars[i] < v_threshold:
            indexes_of_removed_d1.append(i)
            var_count += 1
    for i in range(len(sums)):
        if sums[i] < b_threshold:
            indexes_of_removed_d2.append(i)
            sum_count += 1
            continue
        if pvals[i] < p_threshold:
            indexes_of_removed_d2.append(i)
            pval_count += 1
    nd1 = numpy.delete(nd1, indexes_of_removed_d1, axis=1)
    nd2 = numpy.delete(nd2, indexes_of_removed_d2, axis=1)

    # 重组特征
    d = utils.combine(nd1.tolist(), nd2.tolist())
    X = numpy.array(d)
    y = numpy.array(f)
    # 输入模型
    clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
    clf.fit(X, y)

    # 加载测试数据
    td, tf = reader.readVTestData('./dm_final/data/V1_ECFP4.csv', k_part)
    # 和训练数据一样，对特征进行划分
    td1, td2 = utils.split_v1(td)
    # 删除统计值小于给定阈值的列
    ntd1 = numpy.array(td1)
    ntd2 = numpy.array(td2)
    ntd1 = numpy.delete(ntd1, indexes_of_removed_d1, axis=1)
    ntd2 = numpy.delete(ntd2, indexes_of_removed_d2, axis=1)
    # 重组特征
    td = utils.combine(ntd1.tolist(), ntd2.tolist())
    # 预测测试数据
    r = clf.predict(td)
    # 真实结果与预测结果
    y_true = tf
    y_pred = r.tolist()
    # 评价结果
    utils.show_result(y_true, y_pred)

    return y_true, y_pred

if __name__ == '__main__':
    y_true, y_pred_0 = v1_filter(5, 0.015, 0, 3, 0, True)
    _, y_pred_1 = v1_filter(5, 0.015, 0, 3, 1, False)
    _, y_pred_2 = v1_filter(5, 0.015, 0, 3, 0, False)
    _, y_pred_3 = v1_filter(5, 0.015, 0, 3, 0, True)
    _, y_pred_4 = v1_filter(5, 0.015, 0, 3, 0, True)
    _, y_pred_5 = v1_filter(5, 0.015, 0, 3, 0, True)
    _, y_pred_6 = v1_filter(5, 0.015, 0, 3, 0, True)
    for i in range(len(y_pred_0)):
        c = 0
        if y_pred_0[i] == 1:
            c += 1
        if y_pred_1[i] == 1:
            c += 1
        if y_pred_2[i] == 1:
            c += 1
        if y_pred_3[i] == 1:
            c += 1
        if y_pred_4[i] == 1:
            c += 1
        if y_pred_5[i] == 1:
            c += 1
        if y_pred_6[i] == 1:
            c += 1
        if c >= 1:
            y_pred_0[i] = 1


    utils.show_result(y_true, y_pred_0)