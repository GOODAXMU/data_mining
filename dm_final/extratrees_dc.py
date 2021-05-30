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
# n0: 0类别/1类别
# n2: 2类别/1类别
def dc_filter(b_threshold, v_threshold, p_threshold, k_part, n0, n2):
    # 加载训练数据和测试数据
    d, f, td, tf = reader.readDCData('./dm_final/data/drug_combination.csv', k_part)
    # 控制1在数据集中的比例
    print("before rescale d:", len(d))
    d, f = utils.rescale_dc_data(d, f, n0, n2)
    print("after rescale d:", len(d))
    # 将数值型和类别型数据分开
    d1, d2 = utils.split_dc(d)
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

    # 和训练数据一样，对特征进行划分
    td1, td2 = utils.split_dc(td)
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
    utils.show_dc_result(y_true, y_pred)

    return y_true, y_pred

# 不做任何优化
def dc_optimize_0():
    y_true, y_pred = dc_filter(0, 0, 0, 3, 10000, 10000)
    utils.show_dc_result(y_true, y_pred)
    return

# 总和过滤
def dc_optimize_1():
    for i in range(30):
        y_true, y_pred = dc_filter(i, 0, 0, 3, 10000, 10000)
    return

# 方差过滤
def dc_optimize_2():
    for i in range(50):
        y_true, y_pred = dc_filter(0, i / 200, 0, 3, 10000, 10000)
    return

# 卡方概率过滤
def dc_optimize_3():
    for i in range(50):
        y_true, y_pred = dc_filter(0, 0, i / 100, 3, 10000, 10000)
    return

# 控制012比例为1:1:1，过滤阈值选择上面优化手段得到的最优值
def dc_optimize_4():
    y_true, y_pred = dc_filter(0, 0, 0, 3, 1, 1)
    return

# 多森林投票
def dc_optimeze_5():
    y_true = []
    y_pred_s = []
    for i in range(5):
        y_true_t, y_pred_t = dc_filter(0, 0, 0, 3, 1, 1)
        y_true = y_true_t
        y_pred_s.append(y_pred_t)

    y_pred = []
    for i in range(len(y_pred_s[0])):
        c = 0
        for j in range(len(y_pred_s)):
            if y_pred_s[j][i] == 1:
                c += 1
        if c > 1:
            y_pred.append(1)
        else:
            # 归到哪一类不重要，因为只看recall_1 * precision_1
            y_pred.append(2)
    return

# 用一个模型分开0和其他，再用一个模型分开1，2
def dc_optimeze_6():
    # 分辨0与其他
    y_true_0, y_pred_0 = dc_filter(0, 0, 0, 3, 10000, 1)
    # 分辨1，2
    _, y_pred_12 = dc_filter(0, 0, 0, 3, 0, 1)

    y_pred = []
    y_true = []
    for i in range(len(y_true_0)):
        if y_pred_0[i] == 0:
            continue
        y_pred.append(y_pred_12[i])
        y_true.append(y_true_0[i])
    return


if __name__ == '__main__':
    dc_optimize_0()
    #dc_optimize_1()
    #dc_optimize_2()
    #dc_optimize_3()
    #dc_optimize_4()
    #dc_optimize_5()
    #dc_optimize_6()