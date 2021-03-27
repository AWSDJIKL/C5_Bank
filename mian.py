# -*- coding: utf-8 -*-
'''
主函数
'''
# @Time : 2021/3/27 23:38 
# @Author : LINYANZHEN
# @File : mian.py

import C45
import RandomForest as rf

if __name__ == '__main__':
    # # 导入数据
    data = rf.load_data("bank-full.csv")
    # forest_size = 20
    # clean_data = rf.clean_data(data)
    # # 分割数据集
    # datasets = rf.split_dataset(clean_data, forest_size + 1)
    # # 分开训练集和测试集
    # train_sets = datasets[:-1]
    # test_set = datasets[-1]
    # # 生成随机森林
    # forest = rf.build_random_forest(train_sets, forest_size)
    # # 存档
    # forest.to_json("forest_C45.json")
    # 读档
    forest = rf.RandomForest.json_to_forest("forest_C45.json")
    # 测试准确率
    count = 0
    default_count = 0
    for index, s in data.iterrows():
        # print("age" in s.keys().values)
        classify_result = forest.classify(s)
        if classify_result == s["y"]:
            count += 1
        if not classify_result:
            print("分类失败")
            default_count += 1
        print("预测值：{}  实际值：{}".format(classify_result, s["y"]))
    print("准确率：{}  分类失败率：{}".format(count / len(data), default_count / len(data)))
