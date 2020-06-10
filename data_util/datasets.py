"""
Dataset loading functions
Reference: https://github.com/FlorentF9/DeepTemporalClustering.git
"""
import os
import numpy as np
import argparse
import random


def data_divide(X_train, y_train, data_path):
    index = np.argsort(y_train)
    X_train1 = np.zeros(X_train.shape)
    for i in range(np.size(index)):
        X_train1[i] = X_train[index[i]]
    X_train = X_train1
    y_train = np.sort(y_train)
    count = np.bincount(y_train)
    sum = len(np.unique(y_train))
    count1 = np.zeros((sum,))
    j = 0
    for i in range(np.size(count)):
        if count[i] > 0:
            count1[j] = count[i]
            j = j + 1
    count = count1.astype(np.int32)
    print("each cluster sample number")
    for i in range(np.size(count)):
        print(count[i])
    sum = 0
    for i in range(np.size(count)):
        sum = sum + count[i]
        if i > 0:
            save_path_X = data_path + '/X_train_{}.npy'.format(i + 1)
            save_path_y = data_path + '/y_train_{}.npy'.format(i + 1)
            np.save(save_path_X, X_train[: sum])
            np.save(save_path_y, y_train[: sum])


def data_cluster_divide(x, y):
    index = np.argsort(y)
    x1 = np.zeros(x.shape)
    for i in range(np.size(index)):
        x1[i] = x[index[i]]
    x = x1
    y = np.sort(y)
    count = np.bincount(y)
    sum = len(np.unique(y))
    count1 = np.zeros((sum,))
    if np.size(count) > np.size(count1):
        y = y - 1
    j = 0
    for i in range(np.size(count)):
        if count[i] > 0:
            count1[j] = count[i]
            j = j + 1
    count = count1.astype(np.int32)
    data = []
    sum = 0
    for i in range(np.size(count)):
        sum = sum + count[i]
        if i == 0:
            x_ = x[: count[i]]
            y_ = y[: count[i]]
            data.append((x_, y_))
        else:
            x_ = x[sum - count[i]:sum]
            y_ = y[sum - count[i]:sum]
            data.append((x_, y_))
    return data


def data_cluster_recombination(X_train, y_train, cluster_string, data_path):
    recombination_list = []
    length = len(np.unique(y_train))
    cluster_list = cluster_string.split(',')
    for i in range(len(cluster_list)):
        index = random.sample(range(length), int(cluster_list[i]))
        recombination_list.append(index)

    data = data_cluster_divide(X_train, y_train)
    for i in range(len(recombination_list)):
        for j in range(len(recombination_list[i])):
            if j == 0:
                x = data[recombination_list[i][j]][0]
                y = data[recombination_list[i][j]][1]
            else:
                x = np.concatenate((x, data[recombination_list[i][j]][0]), axis=0)
                y = np.concatenate((y, data[recombination_list[i][j]][1]), axis=0)
        save_path_X = data_path + '/X_train_{}.npy'.format(recombination_list[i])
        save_path_y = data_path + '/y_train_{}.npy'.format(recombination_list[i])
        np.save(save_path_X, x)
        np.save(save_path_y, y)


def data_preprocess(X_train, y_train):
    X_train = X_train.astype(np.float32)
    X_train -= np.mean(X_train, axis=0)
    X_train /= np.std(X_train, axis=0)
    y_train = y_train.astype(np.int32)
    return X_train, y_train


def data_concat(X_train, X_test):
    x = np.concatenate((X_train, X_test), axis=0)
    return x


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path', type=str, default='data/Uwave')
    parser.add_argument('--save_dir', type=str, default='data/Uwave/ll')
    parser.add_argument('--cluster_nums', type=str, default='3,4,5',
                        help='Recombine the data according to the specified cluster')
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    path = args.data_path
    print(os.path.exists(path))
    X_train_path = path + '/X_train.npy'
    y_train_path = path + '/y_train.npy'
    X_train = np.load(X_train_path)
    y_train = np.load(y_train_path)

    X_test_path = path + '/X_test.npy'
    y_test_path = path + '/y_test.npy'
    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)

    X_train = data_concat(X_train, X_test)
    y_train = data_concat(y_train, y_test)
    y_train = y_train.flatten()

    data_divide(X_train, y_train, args.save_dir)

    cluster_nums = args.cluster_nums
    data_cluster_recombination(X_train, y_train, cluster_nums, args.save_dir)
