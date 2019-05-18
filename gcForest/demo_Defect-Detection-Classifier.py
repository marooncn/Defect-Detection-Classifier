#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
MNIST datasets demo for gcforest
Usage:
    define the model within scripts:
        python examples/demo_mnist.py
    get config from json file:
        python examples/demo_mnist.py --model examples/demo_mnist-gc.json
        python examples/demo_mnist.py --model examples/demo_mnist-ca.json
"""
import argparse # 参数
import numpy as np # 科学计算包
import sys # 系统功能调用
# from keras.datasets import mnist # 数据集导入
import pickle # 数据集文件读取
from sklearn.ensemble import RandomForestClassifier # 随机森林分类器
from sklearn.metrics import accuracy_score # metrics 度量 准确率分数
from skimage import io,color

# add the lib path
sys.path.insert(0, "./gcForest/gcForest/lib")  # 0为优先级，最先搜索

from gcforest.gcforest import GCForest
from gcforest.utils.config_utils import load_json


def load_defect_data():
    train_path = []
    train_label = []
    with open("./data/train/label.csv", "r") as file:
        for line in file:
            image_path, image_label = line.split(',')
            train_path.append(image_path)
            train_label.append(image_label.strip('\n'))

    test_path = []
    test_label = []
    with open("./data/test/label.csv", "r") as file:
        for line in file:
            image_path, image_label = line.split(',')
            test_path.append(image_path)
            test_label.append(image_label.strip('\n'))

    coll_train = io.ImageCollection(train_path, load_func=convert_gray)
    coll_test = io.ImageCollection(test_path, load_func=convert_gray)
    coll_train = np.asarray(coll_train)
    coll_test = np.asarray(coll_test)
    train_label = np.asarray(train_label).astype(np.int8)
    test_label = np.asarray(test_label).astype(np.int8)

    X_train = coll_train
    X_test = coll_test
    y_train = train_label
    y_test = test_label

    return (X_train, y_train), (X_test, y_test)

def convert_gray(f):
    rgb=io.imread(f)
    return color.rgb2gray(rgb)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model", type=str, default=None, help="gcfoest Net Model File")
    args = parser.parse_args()
    return args


def get_toy_config():
    config = {}
    ca_config = {}
    ca_config["random_state"] = 0
    ca_config["max_layers"] = 100
    ca_config["early_stopping_rounds"] = 3
    ca_config["n_classes"] = 10
    ca_config["estimators"] = []
    ca_config["estimators"].append(
            {"n_folds": 5, "type": "XGBClassifier", "n_estimators": 10, "max_depth": 5,
             "objective": "multi:softprob", "silent": True, "nthread": -1, "learning_rate": 0.1} )
    ca_config["estimators"].append({"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 10, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": 10, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 5, "type": "LogisticRegression"})
    config["cascade"] = ca_config
    return config


if __name__ == "__main__":
    args = parse_args()
    if args.model is None:
        config = get_toy_config()
    else:
        config = load_json(args.model)

    gc = GCForest(config)
    # If the model you use cost too much memory for you.
    # You can use these methods to force gcforest not keeping model in memory
    # gc.set_keep_model_in_mem(False), default is TRUE.
    gc.set_keep_model_in_mem(False)

    (X_train, y_train), (X_test, y_test) = load_defect_data()
    print('X_train.shape', X_train.shape, 'X_test', X_test.shape, 'y_train', y_train.shape, 'y_test', y_test.shape)
    # X_train, y_train = X_train[:2000], y_train[:2000]
    X_train = X_train[:, np.newaxis, :, :]
    X_test = X_test[:, np.newaxis, :, :]


    X_train_enc = gc.fit_transform(X_train,y_train, X_test=X_test, y_test=y_test)
    # X_enc is the concatenated predict_proba result of each estimators of the last layer of the GCForest model
    # X_enc.shape =
    #   (n_datas, n_estimators * n_classes): If cascade is provided
    #   (n_datas, n_estimators * n_classes, dimX, dimY): If only finegrained part is provided
    # You can also pass X_test, y_test to fit_transform method, then the accracy on test data will be logged when training.
    # X_train_enc, X_test_enc = gc.fit_transform(X_train, y_train, X_test=X_test, y_test=y_test)
    # WARNING: if you set gc.set_keep_model_in_mem(True), you would have to use
    # gc.fit_transform(X_train, y_train, X_test=X_test, y_test=y_test) to evaluate your model.
    
    y_pred = gc.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Test Accuracy of GcForest = {:.2f} %".format(acc * 100))

    # You can try passing X_enc to another classfier on top of gcForest.e.g. xgboost/RF.
    X_test_enc = gc.transform(X_test)
    X_train_enc = X_train_enc.reshape((X_train_enc.shape[0], -1))
    X_test_enc = X_test_enc.reshape((X_test_enc.shape[0], -1))
    X_train_origin = X_train.reshape((X_train.shape[0], -1))
    X_test_origin = X_test.reshape((X_test.shape[0], -1))
    X_train_enc = np.hstack((X_train_origin, X_train_enc))
    X_test_enc = np.hstack((X_test_origin, X_test_enc))
    print("X_train_enc.shape={}, X_test_enc.shape={}".format(X_train_enc.shape, X_test_enc.shape))
    clf = RandomForestClassifier(n_estimators=1000, max_depth=None, n_jobs=-1)
    clf.fit(X_train_enc, y_train)
    y_pred = clf.predict(X_test_enc)
    acc = accuracy_score(y_test, y_pred)
    print("Test Accuracy of Other classifier using gcforest's X_encode = {:.2f} %".format(acc * 100))

    # dump
    with open("test.pkl", "wb") as f:
        pickle.dump(gc, f, pickle.HIGHEST_PROTOCOL)
    # load
    with open("test.pkl", "rb") as f:
        gc = pickle.load(f)
    y_pred = gc.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Test Accuracy of GcForest (save and load) = {:.2f} %".format(acc * 100))
