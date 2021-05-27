import numpy
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2
import feature_selector
import utils
import reader

def extratrees_v1_normal():
    d, f = reader.readVTrainingData('./dm_final/data/V1_ECFP4.csv')

    X = numpy.array(d)
    y = numpy.array(f)

    clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
    clf.fit(X, y)

    td, tf = reader.readVTestData('./dm_final/data/V1_ECFP4.csv')

    r = clf.predict(td)

    y_true = tf
    y_pred = r.tolist()

    utils.show_result(y_true, y_pred)

    return

def extratrees_v1_value_part():
    d, f = reader.readVTrainingData('./dm_final/data/V1_ECFP4.csv')
    d = reader.get_cols(d, 0, 200)

    X = numpy.array(d)
    y = numpy.array(f)

    clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
    clf.fit(X, y)

    td, tf = reader.readVTestData('./dm_final/data/V1_ECFP4.csv')
    td = reader.get_cols(td, 0, 200)

    r = clf.predict(td)

    y_true = tf
    y_pred = r.tolist()

    utils.show_result(y_true, y_pred)

    return

def extratrees_v1_binary_part():
    d, f = reader.readVTrainingData('./dm_final/data/V1_ECFP4.csv')
    d = reader.get_cols(d, 200, -1)

    X = numpy.array(d)
    y = numpy.array(f)

    clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
    clf.fit(X, y)

    td, tf = reader.readVTestData('./dm_final/data/V1_ECFP4.csv')
    td = reader.get_cols(td, 200, -1)

    r = clf.predict(td)

    y_true = tf
    y_pred = r.tolist()

    utils.show_result(y_true, y_pred)

    return

def extratrees_v1_binary_part_with_filter():
    d, f = reader.readVTrainingData('./dm_final/data/V1_ECFP4.csv')
    d = reader.get_cols(d, 200, -1)

    X = numpy.array(d)
    y = numpy.array(f)

    remove_cols = []
    sums = numpy.sum(X, axis=0)
    for i in range(len(sums)):
        if (sums[i] < 5):
            remove_cols.append(i)
    X = numpy.delete(X, remove_cols, axis=1)

    clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
    clf.fit(X, y)

    td, tf = reader.readVTestData('./dm_final/data/V1_ECFP4.csv')
    td = reader.get_cols(td, 200, -1)
    td = numpy.array(td)
    td = numpy.delete(td, remove_cols, axis=1)
    r = clf.predict(td)

    y_true = tf
    y_pred = r.tolist()

    utils.show_result(y_true, y_pred)

    return

def extratrees_v1_normal_with_filter():
    d, f = reader.readVTrainingData('./dm_final/data/V1_ECFP4.csv')

    X = numpy.array(d)
    y = numpy.array(f)

    remove_cols = []
    sums = numpy.sum(X, axis=0)
    for i in range(200, len(sums)):
        if (sums[i] < 5):
            remove_cols.append(i)

    rbs = len(remove_cols)
    print("removed by sum:", rbs)

    plt.figure()
    plt.plot(numpy.arange(0, len(d[0]), 1), sums)
    plt.show()

    vars = numpy.var(X, axis=0)
    for i in range(0, 200):
        if (vars[i] < 0.02):
            remove_cols.append(i)

    print("removed by var:", len(remove_cols) - rbs)

    plt.figure()
    plt.plot(numpy.arange(0, len(d[0]), 1), vars)
    plt.show()

    X = numpy.delete(X, remove_cols, axis=1)

    print(X.shape)
    clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
    clf.fit(X, y)

    td, tf = reader.readVTestData('./dm_final/data/V1_ECFP4.csv')

    td = numpy.array(td)
    td = numpy.delete(td, remove_cols, axis=1)

    r = clf.predict(td)

    y_true = tf
    y_pred = r.tolist()

    utils.show_result(y_true, y_pred)

    return

def extratrees_v1_normal_with_tree_selector():
    d, f = reader.readVTrainingData('./dm_final/data/V1_ECFP4.csv')

    selector = feature_selector.get_tree_selector(d, f)
    d = selector.transform(d)

    X = numpy.array(d)
    y = numpy.array(f)

    clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
    clf.fit(X, y)

    td, tf = reader.readVTestData('./dm_final/data/V1_ECFP4.csv')
    td = selector.transform(td)
    r = clf.predict(td)

    y_true = tf
    y_pred = r.tolist()

    utils.show_result(y_true, y_pred)

    return

# 数据集：V1_ECFP4
# 算法：极限树
# 优化方法：特征选择
# 特征选择算法：SelectKBest
# 特征选择算法的数据：全部特征
# 特征选择算法依据：chi2 卡方检验
# 其他特征选择算法依据：f_classif 样本方差F值；mutual_info_classif 离散类别交互信息
def extratrees_v1_normal_with_KB_selector():
    d, f = reader.readVTrainingData('./dm_final/data/V1_ECFP4.csv')

    selector = feature_selector.get_KB_selector("chi2", d, f, 1500)
    d = selector.transform(d)

    X = numpy.array(d)
    y = numpy.array(f)

    clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
    clf.fit(X, y)

    td, tf = reader.readVTestData('./dm_final/data/V1_ECFP4.csv')
    td = selector.transform(td)
    r = clf.predict(td)

    y_true = tf
    y_pred = r.tolist()

    utils.show_result(y_true, y_pred)

    return

# 数据集：V1_ECFP4
# 算法：极限树
# 优化方法：特征选择
# 特征选择算法：SelectKBest
# 特征选择算法的数据：全部特征
# 特征选择算法依据：mutual_info_classif 离散类别交互信息
# 其他特征选择算法依据：f_classif 样本方差F值；chi2 卡方检验
def extratrees_v1_normal_with_KB_selector_mic():
    d, f = reader.readVTrainingData('./dm_final/data/V1_ECFP4.csv')

    selector = feature_selector.get_KB_selector("mutual_info_classif", d, f, 1500)
    d = selector.transform(d)

    X = numpy.array(d)
    y = numpy.array(f)

    clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
    clf.fit(X, y)

    td, tf = reader.readVTestData('./dm_final/data/V1_ECFP4.csv')
    td = selector.transform(td)
    r = clf.predict(td)

    y_true = tf
    y_pred = r.tolist()

    utils.show_result(y_true, y_pred)

    return

# 数据集：V1_ECFP4
# 算法：极限树
# 优化方法：特征选择
# 特征选择算法：SelectKBest
# 特征选择算法的数据：数值型和类型特征分开
# 特征选择算法依据：chi2 卡方检验
# 其他特征选择算法依据：f_classif 样本方差F值；mutual_info_classif 离散类别交互信息
def extratrees_v1_normal_with_KB_selector_split():
    d, f = reader.readVTrainingData('./dm_final/data/V1_ECFP4.csv')
    d1 = []
    d2 = []
    for i in range(len(d)):
        d1.append(d[i][0 : 200])
    for i in range(len(d)):
        #d2.append(list(map(int, d[i][200 : -1])))
        d2.append(d[i][200 : -1])

    selector1 = feature_selector.get_KB_selector("chi2", d1, f, 150)
    selector2 = feature_selector.get_KB_selector("chi2", d2, f, 1500)

    d1 = selector1.transform(d1).tolist()
    d2 = selector2.transform(d2).tolist()
    d = d1
    for i in range(len(d1)):
        for j in range(len(d2[0])):
            d[i].append(d2[i][j])

    X = numpy.array(d)
    y = numpy.array(f)

    clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
    clf.fit(X, y)

    td, tf = reader.readVTestData('./dm_final/data/V1_ECFP4.csv')
    td1 = []
    td2 = []
    for i in range(len(td)):
        td1.append(td[i][0 : 200])
    for i in range(len(td)):
        #td2.append(list(map(int, td[i][200 : -1])))
        td2.append(td[i][200 : -1])
    td1 = selector1.transform(td1).tolist()
    td2 = selector2.transform(td2).tolist()
    td = td1
    for i in range(len(td1)):
        for j in range(len(td2[0])):
            td[i].append(td2[i][j])
    
    r = clf.predict(td)

    y_true = tf
    y_pred = r.tolist()

    utils.show_result(y_true, y_pred)

    return

# 数据集：V1_ECFP4
# 算法：极限树
# 优化方法：特征选择
# 特征选择算法：SelectKBest
# 特征选择算法的数据：数值型和类型特征分开
# 特征选择算法依据：chi2 卡方检验 和 f_classif 样本方差F值
# 其他特征选择算法依据：mutual_info_classif 离散类别交互信息
def extratrees_v1_normal_with_KB_selector_split_1():
    d, f = reader.readVTrainingData('./dm_final/data/V1_ECFP4.csv')
    d1, d2 = utils.split_v1(d)

    selector1 = feature_selector.get_KB_selector("f_classif", d1, f, 150)
    selector2 = feature_selector.get_KB_selector("chi2", d2, f, 1500)

    d1 = selector1.transform(d1).tolist()
    d2 = selector2.transform(d2).tolist()
    d = d1
    for i in range(len(d1)):
        for j in range(len(d2[0])):
            d[i].append(d2[i][j])

    X = numpy.array(d)
    y = numpy.array(f)

    clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
    clf.fit(X, y)

    td, tf = reader.readVTestData('./dm_final/data/V1_ECFP4.csv')
    td1 = []
    td2 = []
    for i in range(len(td)):
        td1.append(td[i][0 : 200])
    for i in range(len(td)):
        #td2.append(list(map(int, td[i][200 : -1])))
        td2.append(td[i][200 : -1])
    td1 = selector1.transform(td1).tolist()
    td2 = selector2.transform(td2).tolist()
    td = td1
    for i in range(len(td1)):
        for j in range(len(td2[0])):
            td[i].append(td2[i][j])
    
    r = clf.predict(td)

    y_true = tf
    y_pred = r.tolist()

    utils.show_result(y_true, y_pred)

    return

# 先用自己的代码过滤，再用KB过滤
def v1_filter_KB(b_threshold, v_threshold, p_threshold):
    # 加载训练数据
    d, f = reader.readVTrainingData('./dm_final/data/V1_ECFP4.csv')
    # 将数值型和类别型数据分开
    d1, d2 = utils.split_v1(d)
    # 转numpy后计算每列的方差以及每列的和
    nd1 = numpy.array(d1)
    nd2 = numpy.array(d2)
    vars = numpy.var(nd1, axis=0).tolist()
    sums = numpy.sum(nd2, axis=0).tolist()
    # 利用卡方校验计算列与标签之间的关联度
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
    td, tf = reader.readVTestData('./dm_final/data/V1_ECFP4.csv')
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

    return

if __name__ == '__main__':
    #extratrees_v1_normal()
    #extratrees_v1_binary_part()
    #extratrees_v1_binary_part_with_filter()
    #extratrees_v1_value_part()
    #extratrees_v1_normal_with_filter()
    #extratrees_v1_normal_with_feature_select()
    #extratrees_v1_normal_with_KB_selector()
    #extratrees_v1_normal_with_KB_selector_mic()
    #extratrees_v1_normal_with_KB_selector_split()
    #extratrees_v1_normal_with_KB_selector_split_1()
    v1_filter_KB(0, 0.015, 0.1)