# -*- coding: utf-8 -*-
'''
测试代码
'''
# @Time : 2021/3/27 16:00 
# @Author : LINYANZHEN
# @File : test.py
import C45
import RandomForest
import json
import numpy as np

t1 = C45.Tree(None, None, "t1", np.int64(-1))
t2 = C45.Tree(None, None, "t2", None)
t3 = C45.Tree(None, None, "t3", 3)
t4 = C45.Tree(None, None, "t4", 4)
t5 = C45.Tree(None, None, "t5", 5)
print(t1.to_json())

son_nodes1 = {"t2": t2, "t3": t3}
son_nodes2 = {"t4": t4, "t5": t5}
t3.son_nodes = son_nodes2
t1.son_nodes = son_nodes1

trees = [t1, t2, t3, t4, t5]
forest = RandomForest.RandomForest(trees)
forest.to_json("test.json")
# with open("test.json", "w") as file:
#     file.write(forest.to_json())
#
# json.dump(t1.to_dict(), open("test.json", "w"))
#
# t6 = C45.Tree.dict_to_tree(json.load(open("test.json")))
# print(t6.son_nodes)
