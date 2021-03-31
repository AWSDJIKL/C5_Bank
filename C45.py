# -*- coding: utf-8 -*-
'''
C4.5决策树
'''
# @Time : 2021/3/26 15:05 
# @Author : LINYANZHEN
# @File : C45.py

import numpy as np
import json


class Tree:
    def __init__(self, dataset=None, son_nodes=None, feature=None, condition=None, class_result=None):
        # 该节点包含的样本
        self.dataset = dataset
        # 该节点的子节点
        self.son_nodes = son_nodes
        # 该节点选择的特征
        self.feature = feature
        # 该节点选择的分割点（当特征是连续型时生效）
        self.condition = condition
        # 该节点的分类结果（当没有子树的时候生效）
        self.class_result = class_result

    def classify(self, data):
        c = data.keys().values
        if not self.son_nodes:
            # 没有子节点了，直接输出分类结果
            return self.class_result
        if self.feature not in c:
            # 有缺失值，无法分类
            # print("有缺失值，无法分类")
            return None
        # 判断该节点特征的数据类型
        x = data[self.feature]
        if type(x) in (int, float):
            # 是连续型
            if x <= self.condition:
                return self.son_nodes["<=" + str(self.condition)].classify(data)
            else:
                return self.son_nodes[">" + str(self.condition)].classify(data)
        else:
            # 是离散型
            if str(x) not in self.son_nodes.keys():
                # print("该树无法对此样本分类")
                return None
            return self.son_nodes[str(x)].classify(data)

    def to_dict(self):
        tree_dict = {}
        son_nodes = {}
        if self.son_nodes:
            for key, node in self.son_nodes.items():
                son_nodes[key] = node.to_dict()
            tree_dict["son_nodes"] = son_nodes
        else:
            tree_dict["son_nodes"] = None
        tree_dict["feature"] = self.feature
        if type(self.condition) == np.int64:
            # np.int64转int需要先转str再转int
            tree_dict["condition"] = int(str(self.condition))
        else:
            tree_dict["condition"] = self.condition
        tree_dict["class_result"] = self.class_result
        return tree_dict

    def to_json(self):
        d = self.to_dict()
        return json.dumps(d)

    @classmethod
    def json_to_tree(cls, json_str):
        d = json.loads(json_str)
        return Tree.dict_to_tree(d)

    @classmethod
    def dict_to_tree(cls, tree_dict):
        if tree_dict["son_nodes"]:
            son_node = {}
            for key, node in tree_dict["son_nodes"].items():
                son_node[key] = Tree.dict_to_tree(node)
            if tree_dict["condition"]:
                tree_dict["condition"] = np.int64(tree_dict["condition"])
            return Tree(None, son_node, tree_dict["feature"], tree_dict["condition"], tree_dict["class_result"])
        else:
            if tree_dict["condition"]:
                tree_dict["condition"] = np.int64(tree_dict["condition"])
            return Tree(None, None, tree_dict["feature"], tree_dict["condition"], tree_dict["class_result"])


# 信息熵（通用）
def entropy_y(y):
    y = list(y)
    y_types = set(y)
    entropy = 0.0
    # 信息熵=（概率*log2概率）累加
    for i in y_types:
        p_yi = y.count(i) / len(y)
        entropy += -(p_yi * np.log2(p_yi))
    return entropy


# 连续型变量条件熵
def continuous_con_ent(y, x):
    div_point = []
    ent = []
    # 找出所有的分割点
    for i in range(len(y) - 1):
        if y.iloc[i + 1] != y.iloc[i]:
            div_point.append(x.iloc[i])
    # 对于每一个分割点都计算条件熵
    for i in div_point:
        # 根据分割点分开2部分
        smaller = y[x <= i]
        bigger = y[x > i]
        # 分别计算分类后的信息熵,然后乘以概率，累加得到条件熵
        smaller_ent = entropy_y(smaller)
        smaller_p = len(x[x <= i]) / len(y)
        bigger_ent = entropy_y(bigger)
        bigger_p = len(x[x > i]) / len(y)
        ent.append(smaller_p * smaller_ent + bigger_p * bigger_ent)
    return ent


# 连续性变量的信息熵
def continuous_entropy_x(y, x):
    div_point = []
    entropy = []
    # print(x)
    # 找出分割点
    for i in range(len(y) - 1):
        if y.iloc[i + 1] != y.iloc[i]:
            div_point.append(x.iloc[i])
    # 计算所有分割点对应的信息熵
    # print(div_point)
    for j in div_point:
        # 计算分割后2边的样本个数
        smaller_x_count = np.sum(x <= j)
        bigger_x_count = np.sum(x > j)
        total = smaller_x_count + bigger_x_count
        smaller_x_p = smaller_x_count / total
        smaller_ent = 0
        bigger_ent = 0
        if smaller_x_p != 0:
            smaller_ent = smaller_x_p * np.log2(smaller_x_p)
        bigger_x_p = bigger_x_count / total
        if bigger_x_p != 0:
            bigger_ent = bigger_x_p * np.log2(bigger_x_p)
        entropy.append(-(smaller_ent + bigger_ent))
    return entropy


# 连续变量的信息增益率
def continuous_gain_rate(y, x):
    # 记录分割点
    d = []
    # 记录每个分割点对应的信息增益
    f = []
    for i in range(len(y) - 1):
        # 出现类别不同，即这里可以是一个分割点
        # 因为在同一类里进行分割并不提高信息增益率
        if y.iloc[i + 1] != y.iloc[i]:
            d.append(x.iloc[i])
    # 计算条件信息熵
    con_ent = continuous_con_ent(y, x)
    for j in con_ent:
        # 信息增益 = 分类属性的信息熵 - 条件信息熵
        f.append(entropy_y(y) - j)
    # 计算信息增益率 = 条件熵 / 信息熵
    best_point = 0
    best_gainrate = 0
    entropy_x = continuous_entropy_x(y, x)
    for k in range(len(con_ent)):
        # 信息增益率 = 信息增益 / 特征的信息熵
        if entropy_x[k] == 0:
            continue
        now_gainrate = f[k] / entropy_x[k]
        if best_gainrate < now_gainrate:
            best_gainrate = now_gainrate
            best_point = d[k]
    return best_point, best_gainrate


# 离散型变量条件熵
def discrete_con_ent(y, x):
    x_types = set(list(x))
    con_entropy = 0
    for i in x_types:
        y_xi = y[x == i]
        p_y_xi = len(y_xi) / len(y)
        con_entropy += -(p_y_xi * entropy_y(y_xi))
    return con_entropy


# 离散型变量的信息熵
def dicrete_entropy_x(y, x):
    x_types = set(list(x))
    entropy = 0
    for i in x_types:
        y_xi = y[x == i]
        p_y_xi = len(y_xi) / len(y)
        entropy += -(p_y_xi * np.log2(p_y_xi))
    return entropy


# 离散型变量的信息增益率
def discrete_gain_rate(y, x):
    ent_x = dicrete_entropy_x(y, x)
    if ent_x == 0:
        return 0
    ent_y = entropy_y(y)
    con_ent = discrete_con_ent(y, x)
    gain_rate = (ent_y - con_ent) / ent_x
    return gain_rate


# 生成C4.5决策树
def build_decision_tree(dataset, fields):
    print("子节点数据集大小：", len(dataset))
    y = dataset["y"]
    if len(set(list(y))) == 1:
        # 已经完全分类
        node = Tree(dataset, class_result=y.iloc[0])
        return node
    if len(fields) == 0:
        # 特征池已空
        # 把数据集中数量最多的类作为该节点的分类结果
        node = Tree(dataset, class_result=y.value_counts().index[0])
        return node
    best_gain_rate = 0
    best_feature = ""
    # 最佳分割点（对于连续型变量）
    best_point = 0
    # 在特征池中选取最好的分类特征
    for field in fields:
        x = dataset[field]
        # 判断特征类型
        if x.dtype == np.int64:
            # 是连续型
            dataset.sort_values(by=field, inplace=True)
            # 重新获取
            x = dataset[field]
            y = dataset["y"]
            div_point, gain_rate = continuous_gain_rate(y, x)
            # 记录下最好的分类特征
            if best_gain_rate < gain_rate:
                best_feature = field
                best_point = div_point
                best_gain_rate = gain_rate
        else:
            # 是离散型
            gain_rate = discrete_gain_rate(y, x)
            if best_gain_rate < gain_rate:
                best_feature = field
                best_gain_rate = gain_rate
    if best_gain_rate == 0:
        # 无法根据剩余特征进行分类
        node = Tree(dataset, class_result=y.value_counts().index[0])
        return node
    # 判断最优特征的数据类型
    x = dataset[best_feature]
    fields.remove(best_feature)
    son_nodes = {}
    if x.dtype == np.int64:
        # 是连续型
        # 对数据集进行分割
        smaller_set = dataset[x <= best_point].copy(deep=True)
        bigger_set = dataset[x > best_point].copy(deep=True)
        smaller_node = build_decision_tree(smaller_set, fields[:])
        bigger_node = build_decision_tree(bigger_set, fields[:])
        son_nodes["<=" + str(best_point)] = smaller_node
        son_nodes[">" + str(best_point)] = bigger_node
        node = Tree(dataset, son_nodes=son_nodes, feature=best_feature, condition=best_point)
        return node
    else:
        # 是离散型
        x_types = set(list(x))
        # 对数据进行分割
        for i in x_types:
            sub_dataset = dataset[x == i].copy(deep=True)
            son_node = build_decision_tree(sub_dataset, fields[:])
            son_nodes[str(i)] = son_node
        node = Tree(dataset, son_nodes=son_nodes, feature=best_feature)
        return node
