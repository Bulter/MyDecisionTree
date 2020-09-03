import numpy as np


class MyEntropy:
    # def __init__(self, pd_data):
    #     """
    #     自定义类，可计算输入数据集的信息熵，信息增益，信息增益率
    #     :param pd_data: pandas数据，最后一列为类别标签
    #     """
    #     self.data = pd_data
    #     self.data_len = len(pd_data)
    #
    #     self.class_dict = {}
    #     self.total_ent = self.init()

    def id3(self, pd_data):
        """
        使用信息增益作为属性划分的指标
        :param pd_data: pandas格式的训练数据，最后一列为类别标签
        :return:
        """
        optimal_attr, attr_entropy = self.find_optimal_attr(pd_data)
        print(optimal_attr, attr_entropy)

        if len(set(pd_data.iloc[:, -1])) == 1:
            return pd_data.iloc[:, -1][0]

    def find_optimal_attr(self, pd_data):
        attr_entropy_dict = {}
        total_entropy = self.calc_entropy(pd_data.iloc[:, -1])
        for c in pd_data.columns[:-1]:
            attr_entropy_dict[c] = total_entropy + self.calc_attribute_entropy(pd_data, c)
        attribute, attr_entropy = sorted(attr_entropy_dict.items(), key=lambda it: it[1])[-1]
        return attribute, attr_entropy

    def calc_attribute_entropy(self, pd_data, attribute):
        attr_dict = {}
        data = pd_data[attribute]
        data_len = len(data)
        for attr_v in data:
            if attr_v in attr_dict:
                attr_dict[attr_v] += 1
            else:
                attr_dict[attr_v] = 1
        attr_entropy = 0
        for attr_v in attr_dict:
            attr_data = pd_data[pd_data[attribute] == attr_v].iloc[:, -1]
            attr_prob = attr_dict[attr_v] / data_len
            attr_entropy -= attr_prob * self.calc_entropy(attr_data)
        return attr_entropy

    def calc_entropy(self, data):
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


if __name__ == '__main__':
    import pandas as pd

    # data_path = r"C:\Users\11\PycharmProjects\data\watermelon_3.csv"
    # pd_data = pd.read_csv(data_path)
    # data = pd_data.iloc[:, [1, 2, 3, 4, 5, 6, 9]]
    data_path = r"C:\Users\11\PycharmProjects\data\watermelon_2.csv"
    pd_data = pd.read_csv(data_path)
    data = pd_data.iloc[:, 1:]
    result = MyEntropy().id3(data)
    print(result)
