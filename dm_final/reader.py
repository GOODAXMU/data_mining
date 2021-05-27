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
            if i > 10:
                break
            r.append(list(map(float, row[1 : len(row) - 1])))
            flag.append(int(row[len(row) - 1]))
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
            if i > 10:
                break
            r.append(list(map(float, row[1 : len(row) - 1])))
            flag.append(int(row[len(row) - 1]))
    return r, flag

def readVTrainingData(file):
    r = []
    flag = []
    with open(file, 'r') as f:
        reader = csv.reader(f)
        i = 0
        for row in reader:
            i += 1
            if i == 1 or i % 3 == 0:
                continue
            r.append(list(map(float, row[0 : len(row) - 1])))
            flag.append(int(row[len(row) - 1]))
    return r, flag

def readVTestData(file):
    r = []
    flag = []
    with open(file, 'r') as f:
        reader = csv.reader(f)
        i = 0
        for row in reader:
            i += 1
            if i == 1 or i % 3 != 0:
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
    return arr

def get_cols(arr, s, e):
    r = []
    for i in range(len(arr)):
        r.append(arr[i][s : e])
    return r

if __name__ == '__main__':
    d, f = readDCTrainingData('./dm_final/data/drug_combination.csv')