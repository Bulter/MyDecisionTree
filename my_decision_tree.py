from collections import Counter

import numpy as np


class MyDecisionTree:
    def __init__(self):
        self.attr_dict = None

    def generate_tree(self, pd_data, labels, idea='ID3'):
        """
        生成决策树（字典形式）
        :param pd_data: pandas格式的训练数据，最后一列为类别标签
        :param labels: 列表格式数据，与pd_data列名对应，为特征名称
        :param labels: 字符串数据，可选有['ID3', 'C4.5']，生成决策树的方法
        :return: 字典格式数据
        """
        if self.attr_dict is None:
            self.attr_dict = {}
            for attr in labels:
                self.attr_dict[attr] = set(pd_data[attr])

        # 如果标签列只有一个分类结果，则结束递归
        class_dict = dict(Counter(pd_data.iloc[:, -1]))
        if len(class_dict) == 1:
            return list(class_dict.keys())[0]

        # 寻找最适合作为划分的属性
        optimal_attr = self.find_optimal_attr(pd_data, labels, idea)
        attr_dict = dict(Counter(pd_data[optimal_attr]))
        labels.remove(optimal_attr)

        # 如果属性列表为空或属性列的值都相同，则返回此时样本集合中数目最多的类
        if len(labels) == 0 or len(attr_dict) == 1:
            return max(class_dict, key=lambda k: class_dict.get(k))

        # 如果属性列不为空且取值唯一，则按不同属性将样本集合拆分为一个个小的子集并剔除当前属性列，开始递归
        # 注意此处遍历属性取值的集合应为最原始训练数据属性的取值，避免漏掉一些递归到这里属性值被剔除的情况（这种情况下分类结果为当前样本中最多的类）
        my_tree_dict = {optimal_attr: {}}
        for attr_value in self.attr_dict[optimal_attr]:
            sub_labels = labels.copy()
            sub_pd_data = pd_data[pd_data[optimal_attr] == attr_value].drop(labels=optimal_attr, axis=1)
            if len(sub_pd_data) == 0:
                # 如果子集为空，则选择当前样本集中样本最多的类作为分类结果
                my_tree_dict[optimal_attr][attr_value] = max(class_dict, key=lambda k: class_dict.get(k))
            else:
                my_tree_dict[optimal_attr][attr_value] = self.generate_tree(sub_pd_data, sub_labels, idea)

        return my_tree_dict

    def find_optimal_attr(self, pd_data, labels, idea):
        """
        寻找最佳划分属性
        :param pd_data: pandas格式的训练数据，最后一列为类别标签
        :param labels: 列表格式数据，与pd_data列名对应，为特征名称
        :param labels: 字符串数据，可选有['ID3', 'C4.5']，生成决策树的方法
        :return: 字典格式数据
        """
        if idea == 'ID3':
            return self.find_optimal_attr_by_entropy(pd_data, labels)
        elif idea == 'C4.5':
            return self.find_optimal_attr_by_entropy_ratio(pd_data, labels)
        else:
            raise ("argument idea must is 'ID3' or 'C4.5'")

    def find_optimal_attr_by_entropy_ratio(self, pd_data, labels):
        """
        找到最优划分属性：先计算出每个属性的信息增益和信息增益率，再从信息增益高于平均值的属性中找到信息增益率最高的属性
        :param pd_data: pandas格式的训练数据，最后一列为类别标签
        :param labels: 列表格式数据，与pd_data列名对应，为特征名称
        :return attribute: 最优的属性
        """
        attr_entropy_gain_dict = {}
        attr_entropy_gain_ratio_dict = {}
        total_entropy = self.calc_entropy(pd_data.iloc[:, -1])
        for attr in labels:
            entropy = total_entropy + self.calc_attribute_entropy(pd_data, attr)
            attr_entropy_gain_dict[attr] = entropy
            attr_data_len = len(pd_data[attr])
            iv = 0 - np.sum([(i / attr_data_len) * np.log2(i / attr_data_len) for i in Counter(pd_data[attr]).values()])
            attr_entropy_gain_ratio_dict[attr] = entropy / iv

        entropy_mean = np.mean(list(attr_entropy_gain_dict.values()))
        attr_dict = {k: attr_entropy_gain_ratio_dict[k] for k, v in attr_entropy_gain_dict.items() if v >= entropy_mean}
        attribute = max(attr_dict, key=lambda k: attr_dict.get(k))
        print("labels", labels, "mean", entropy_mean, "attribute", attribute)
        print("gain", attr_entropy_gain_dict)
        print("gain_ratio", attr_entropy_gain_ratio_dict)
        return attribute

    def find_optimal_attr_by_entropy(self, pd_data, labels):
        """
        找到最优划分属性：计算每个属性的信息增益作为指标确定最优划分属性
        :param pd_data: pandas格式的训练数据，最后一列为类别标签
        :param labels: 列表格式数据，与pd_data列名对应，为特征名称
        :return attribute: 最优的属性
        """
        attr_entropy_gain_dict = {}
        total_entropy = self.calc_entropy(pd_data.iloc[:, -1])
        for attr in labels:
            attr_entropy_gain_dict[attr] = total_entropy + self.calc_attribute_entropy(pd_data, attr)
        attribute = max(attr_entropy_gain_dict, key=lambda k: attr_entropy_gain_dict.get(k))
        return attribute

    def calc_attribute_entropy(self, pd_data, attribute):
        attr_amount_dict = {}
        data = pd_data[attribute]
        data_len = len(data)
        for attr_v in data:
            if attr_v not in attr_amount_dict:
                attr_amount_dict[attr_v] = 0
            attr_amount_dict[attr_v] += 1
        attr_entropy = 0
        for attr_v in attr_amount_dict:
            attr_data = pd_data[pd_data[attribute] == attr_v].iloc[:, -1]
            attr_prob = attr_amount_dict[attr_v] / data_len
            attr_entropy -= attr_prob * self.calc_entropy(attr_data)
        return attr_entropy

    def calc_entropy_old(self, data):
        """
        计算输入数据集合的信息熵
        :param data: 列表数据
        :return:
        """
        class_dict = {}
        data_len = len(data)
        for v in data:
            if v not in class_dict:
                class_dict[v] = 0
            class_dict[v] += 1
        entropy = 0
        for k in class_dict:
            prob = class_dict[k] / data_len
            entropy -= prob * np.log2(prob)
        return entropy

    def calc_entropy(self, data):
        """
        计算输入数据集合的信息熵
        :param data: 列表数据
        :return:
        """
        entropy = 0
        data_len = len(data)
        amount_dict = dict(Counter(data))
        for k in amount_dict:
            entropy -= (amount_dict[k] / data_len) * np.log2(amount_dict[k] / data_len)
        return entropy


if __name__ == '__main__':
    import pandas as pd
    from utils import createPlot

    # data_path = r"C:\Users\11\PycharmProjects\data\watermelon_3.csv"
    # pd_data = pd.read_csv(data_path)
    # data = pd_data.iloc[:, [1, 2, 3, 4, 5, 6, 9]]
    data_path = r"C:\Users\11\PycharmProjects\data\watermelon_2.csv"
    pd_data = pd.read_csv(data_path)
    data = pd_data.iloc[:, 1:]
    labels = list(data.columns[:-1])
    # print(data)
    print("Origin labels", labels)
    result = MyDecisionTree().generate_tree(data, labels, idea='ID3')
    # result = MyDecisionTree().generate_tree(data, labels, idea='C4.5')
    createPlot(result)
