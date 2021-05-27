import csv
import random

def readDCTrainingData(file):
    r = []
    flag = []
    with open(file, 'r') as f:
        reader = csv.reader(f)
        i = 0
        for row in reader:
            i += 1
            if i == 1 or i % 3 == 0:
                continue
            # 非调试情况下请删掉！！！！！！
            if i > 10:
                break
            r.append(list(map(float, row[1 : len(row) - 1])))
            flag.append(int(float(row[len(row) - 1])))
    return r, flag

def readDCTestData(file):
    r = []
    flag = []
    with open(file, 'r') as f:
        reader = csv.reader(f)
        i = 0
        for row in reader:
            i += 1
            if i == 1 or i % 3 != 0:
                continue
            # 非调试情况下请删掉！！！！！！
            if i > 10:
                break
            r.append(list(map(float, row[1 : len(row) - 1])))
            flag.append(int(float(row[len(row) - 1])))
    return r, flag

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
    readDCTrainingData('./dm_final/data/drug_combination.csv')
    readDCTestData('./dm_final/data/drug_combination.csv')