# -*- coding: utf-8 -*-
'''
随机森林
'''
# @Time : 2021/3/27 10:06 
# @Author : LINYANZHEN
# @File : RandomForest.py

import C45
import pandas as pd
import random
import json


class RandomForest:
    def __init__(self, trees):
        self.trees = trees

    def classify(self, data):
        prevision = []
        for tree in self.trees:
            classify_result = tree.classify(data)
            if classify_result:
                prevision.append(classify_result)
        if len(prevision) > 0:
            return max(prevision, key=prevision.count)
        else:
            print("分类失败")
            return None

    def to_json(self, save_path):
        result = []
        for tree in self.trees:
            result.append(tree.to_dict())
        return json.dump(result, open(save_path, "w"))

    @classmethod
    def json_to_forest(cls, json_path):
        trees = json.load(open(json_path))
        for i in range(len(trees)):
            trees[i] = C45.Tree.dict_to_tree(trees[i])
        return RandomForest(trees)


# 分割数据集
def split_dataset(dataset, size):
    datasets = []
    # 根据森林大小生成对应数量的数据集
    for i in range(size):
        datasets.append(dataset.sample(frac=1.0, replace=True))
    return datasets


# 生成随机森林
def build_random_forest(datasets, forest_size):
    columns = datasets[0].columns[:-1].tolist()
    trees = []
    for i in range(forest_size):
        print("正在生成第{}棵决策树".format(i + 1))
        # 随机生成特征池
        fields = random.sample(columns, random.randint(5, 10))
        # fields = random.sample(columns, 8)
        tree = C45.build_decision_tree(datasets[i], fields)
        trees.append(tree)
    forest = RandomForest(trees)
    return forest


# 读取数据集
def load_data(path):
    data = pd.read_table(path, delimiter=";")
    return data


# 清洗数据
def clean_data(data):
    # 去掉带有unknown的行（数据清洗）
    for i in data.columns:
        row = data[data[i] == "unknown"].index
        data = data.drop(row)
    return data
