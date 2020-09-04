import os
import sys
import numpy as np
from sklearn.model_selection import KFold
from ecoc_tools import ecoc_tool
import subprocess

def gen_data(fold, images, lables):
    newshape = (fold[0].shape[0],) + images.shape[1:]
    train_images = np.empty(newshape)

    newshape = (fold[0].shape[0],) + lables.shape[1:]
    train_lables = np.empty(newshape)

    newshape = (fold[1].shape[0],) + images.shape[1:]
    test_images = np.empty(newshape)

    newshape = (fold[1].shape[0],) + lables.shape[1:]
    test_lables = np.empty(newshape)

    pos = 0
    for part in fold[0]:
        train_images[pos] = images[part]
        train_lables[pos] = lables[part]
        pos += 1

    pos = 0
    for part in fold[1]:
        test_images[pos] = images[part]
        test_lables[pos] = lables[part]
        pos += 1

    return train_images, train_lables, test_images, test_lables

def gen_data_train_data(downsmaple, images, lables):
    newshape = (downsmaple.shape[0],) + images.shape[1:]
    train_images = np.empty(newshape)

    newshape = (downsmaple.shape[0],) + lables.shape[1:]
    train_lables = np.empty(newshape)

    pos = 0
    for part in downsmaple:
        train_images[pos] = images[part]
        train_lables[pos] = lables[part]
        pos += 1

    return train_images, train_lables

def main():

    number_of_folds = 3
    umberla_path = "/home/sindri/NicksPlayGround/ml/deeplearning_ecoc/data/dropout_testdata_70"

    tool = ecoc_tool(np.loadtxt('/home/sindri/NicksPlayGround/ml/deeplearning_ecoc/10x10', delimiter=','))

    if not os.path.exists(umberla_path):

        os.makedirs(os.path.join(umberla_path,"results"))

        pos = 0
        while( pos < number_of_folds):
            os.makedirs(os.path.join(umberla_path,"test_1",str(pos)))
            os.makedirs(os.path.join(umberla_path,"test_2",str(pos)))
            os.makedirs(os.path.join(umberla_path,"test_3",str(pos)))
            pos += 1

    images = np.load("/home/sindri/NicksPlayGround/ml/deeplearning_ecoc/cifar/cifar_images.npy")
    labels = np.load("/home/sindri/NicksPlayGround/ml/deeplearning_ecoc/cifar/cifar_ecoc_labels.npy")

    folds = KFold(n_splits=int(number_of_folds), random_state=34, shuffle=True).split(images)

    pos = 0
    for fold in folds:

        train_images, train_lables, test_images, test_lables = gen_data(fold, images, labels)

        np.save(os.path.join(umberla_path,"test_1",str(pos), "train_images"), train_images)
        np.save(os.path.join(umberla_path,"test_1",str(pos), "train_lables"), train_lables)
        np.save(os.path.join(umberla_path,"test_1",str(pos), "test_images"), test_images)
        np.save(os.path.join(umberla_path,"test_1",str(pos), "test_lables"), test_lables)

        downsmaple = tool.random_dropout(fold[0].shape[0], 3*fold[0].shape[0]/10, 42)

        train_images, train_lables = gen_data_train_data(downsmaple, images, labels)

        np.save(os.path.join(umberla_path,"test_2",str(pos), "train_images"), train_images)
        np.save(os.path.join(umberla_path,"test_2",str(pos), "train_lables"), train_lables)

        count = 0
        while(count < 10):

            downsmaple = tool.random_dropout(fold[0].shape[0], (3*fold[0].shape[0]/10), count)
            train_images, train_lables = gen_data_train_data(downsmaple, images, labels)

            np.save(os.path.join(umberla_path,"test_3", str(pos), "images_{}".format(count)), train_images)
            np.save(os.path.join(umberla_path,"test_3", str(pos), "lables_{}".format(count)), train_lables[:,count])

            count += 1

        pos += 1


if __name__ == "__main__":
    main()
