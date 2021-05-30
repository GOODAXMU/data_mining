import csv
import random

def readDCData(file, k):
    # 训练集特征
    d = []
    # 训练集标签
    flag = []
    # 测试集特征
    td = []
    # 测试集标签
    tf = []
    with open(file, 'r') as f:
        reader = csv.reader(f)
        i = 0
        for row in reader:
            i += 1
            if i == 1:
                continue
            temp_d = []
            for j in range(882, 1915):
                temp_d.append(float(row[j]))
            for j in range(2796, 4806):
                temp_d.append(float(row[j]))
            for j in range(1, 882):
                temp_d.append(float(row[j]))
            for j in range(1915, 2796):
                temp_d.append(float(row[j]))
            if i % k == 0:
                td.append(temp_d)
                tf.append(float(row[-1]))
            else:
                d.append(temp_d)
                flag.append(float(row[-1]))
    c_0 = 0
    c_1 = 0
    c_2 = 0

    return d, flag, td, tf

# k表示讲数据集分为k份，训练数据排除其中一份，需要与加载测试数据时的k值相等
def readVTrainingData(file, k, do_shuffle=False):
    r = []
    flag = []
    with open(file, 'r') as f:
        reader = csv.reader(f)
        i = 0
        for row in reader:
            i += 1
            if i == 1 or i % k == 0:
                continue
            r.append(list(map(float, row[0 : len(row) - 1])))
            flag.append(int(row[len(row) - 1]))
    if do_shuffle:
        r, flag = shuffle(r, flag)

    return r, flag

# 将标签为1的数据等分为两份，0不分，第n份1与全部0，n 属于 {0, 1}，当n大于1，选取全部
def readVTrainingData_2(file, k, n, do_shuffle=False):
    r = []
    flag = []
    with open(file, 'r') as f:
        reader = csv.reader(f)
        i = 0
        for row in reader:
            i += 1
            if i == 1 or i % k == 0:
                continue
            r.append(list(map(float, row[0 : len(row) - 1])))
            flag.append(int(row[len(row) - 1]))
    if do_shuffle:
        r, flag = shuffle(r, flag)
    fr = []
    fflag = []
    for i in range(len(r)):
        if (flag[i] == 1 and i % 2 == n):
            continue
        fr.append(r[i])
        fflag.append(flag[i])
    print(len(fr), len(fr[0]))
    return fr, fflag

# k表示讲数据集分为k份，测试数据选取其中一份，需要与加载训练数据时的k值相等
def readVTestData(file, k):
    r = []
    flag = []
    with open(file, 'r') as f:
        reader = csv.reader(f)
        i = 0
        for row in reader:
            i += 1
            if i == 1 or i % k != 0:
                continue
            r.append(list(map(float, row[0 : len(row) - 1])))
            flag.append(int(row[len(row) - 1]))
    return r, flag

def shuffle(arr, y):
    n = len(arr)
    for i in range(len(arr)):
        t0 = random.randint(i, n - 1)
        t1 = arr[t0]
        arr[t0] = arr[i]
        arr[i] = t1
        t1 = y[t0]
        y[t0] = y[i]
        y[i] = t1
    return arr, y

def get_cols(arr, s, e):
    r = []
    for i in range(len(arr)):
        r.append(arr[i][s : e])
    return r

if __name__ == '__main__':
    readDCData('./dm_final/data/drug_combination.csv')