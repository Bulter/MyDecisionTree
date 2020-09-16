from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold


class MyNaiveBayes:
    def __init__(self):
        # 用来存储连续型随机变量的属性名
        self._continue_attr_set = set()

        # 初始化字典用来存储计算指标
        self._labels_dict = {}
        self._prior_probability_dict = {}
        self._conditional_probability_dict = {}

    def my_max_likelihood_estimation(self, pd_data, labels):
        """
        极大似然估计
        :param pd_data: pandas格式的训练数据，最后一列为类别标签
        :param labels: 列表格式数据，与pd_data列名对应，为特征名称
        :return:
        """
        data_len = len(pd_data)
        for label in labels:
            self._labels_dict[label] = list(Counter(pd_data[label]))

        # 1.计算样本的先验概率
        for k, v in Counter(pd_data.iloc[:, -1]).items():
            self._prior_probability_dict[k] = v / data_len

        # 2.计算样本的条件概率（或者说似然度）
        for class_v in self._prior_probability_dict:
            y_dict = {}
            y_data = pd_data[pd_data.iloc[:, -1] == class_v]
            y_data_len = len(y_data)
            for attr in labels:
                y_dict[attr] = {}
                # 如果是连续型值，则假定取值符合正态分布，计算出均值和样本方差存进字典
                if y_data[attr].dtypes != np.object:
                    self._continue_attr_set.add(attr)
                    y_dict[attr]['mean'] = np.mean(y_data[attr])
                    y_dict[attr]['std'] = np.std(y_data[attr], ddof=1)
                    continue
                # 如果是离散型变量，则使用样本空间的频率作为概率
                y_counter = Counter(y_data[attr])
                for attr_v in self._labels_dict[attr]:
                    y_dict[attr][attr_v] = y_counter.get(attr_v, 0) / y_data_len
            self._conditional_probability_dict[class_v] = y_dict

    def my_bayes_estimation(self, pd_data, labels, lam=1):
        """
        贝叶斯估计，默认lambda=1即为拉普拉斯平滑
        :param pd_data: pandas格式的训练数据，最后一列为类别标签
        :param labels: 列表格式数据，与pd_data列名对应，为特征名称
        :param lam: 正数，当lambda=0即为极大似然估计
        :return:
        """
        data_len = len(pd_data)
        for label in labels:
            self._labels_dict[label] = list(Counter(pd_data[label]))

        # 1.计算样本的先验概率
        class_counter = Counter(pd_data.iloc[:, -1])
        for k, v in class_counter.items():
            self._prior_probability_dict[k] = (v + lam) / (data_len + lam * len(class_counter))

        # 2.计算样本的条件概率（或者说似然度）
        for class_v in self._prior_probability_dict:
            y_dict = {}
            y_data = pd_data[pd_data.iloc[:, -1] == class_v]
            y_data_len = len(y_data)
            for attr in labels:
                y_dict[attr] = {}
                # 如果是连续型值，则假定取值符合正态分布，计算出均值和样本方差存进字典
                if y_data[attr].dtypes != np.object:
                    self._continue_attr_set.add(attr)
                    y_dict[attr]['mean'] = np.mean(y_data[attr])
                    y_dict[attr]['std'] = np.std(y_data[attr], ddof=1)
                    continue
                # 如果是离散型变量，则使用样本空间的频率作为概率
                y_counter = Counter(y_data[attr])
                for attr_v in self._labels_dict[attr]:
                    y_dict[attr][attr_v] = (y_counter.get(attr_v, 0) + lam) / (y_data_len + lam * len(y_counter))
            self._conditional_probability_dict[class_v] = y_dict

    def train(self, pd_data, labels, name='MLE'):
        """
        对传入的数据进行训练得到朴素贝叶斯分类器
        :param pd_data: pandas格式的训练数据，最后一列为类别标签
        :param labels: 列表格式数据，与pd_data列名对应，为特征名称
        :param name: 字符串数据，可选有['MLE', 'Bayes']，生成决策树的方法
        :return: 字典格式数据
        """
        # 0.前期准备
        # self.__init__()

        if name == 'MLE':
            self.my_max_likelihood_estimation(pd_data, labels)
        elif name == 'Bayes':
            self.my_bayes_estimation(pd_data, labels)
        else:
            raise ("argument name must is 'MLE' or 'Bayes'")

        print(self._labels_dict)
        print(self._prior_probability_dict)
        print(self._conditional_probability_dict)

    def predict(self, pd_data):
        """
        根据训练数据得到的分类器对传入的数据进行分类，
        由贝叶斯公式得 P(y|x) = P(x|y)P(y)/P(x)，因为对于同一个 x P(x)一样，所以 P(y|x) 正比于 P(y)P(x|y),
        根据朴素贝叶斯的思想，分类器采用了“属性条件独立性假设”（对已知类别【注一】，假设所有属性相互独立），
        即 P(x|y) = P(x1|y)P(x2|y)P(x3|y)...，所以 P(y|x) 正比于 P(y)P(x1|y)P(x2|y)P(x3|y)...

        注一：假设只是简化了贝叶斯公式的分子，并不足以简化其分母，所以分母只能用全概率公司求，尽管此处也不涉及分母
        :param pd_data: pandas格式的测试数据，列名属于训练数据传入的labels子集
        :return:
        """
        # 3.计算传入样本的 P(y|x)
        prob_list = []
        labels = pd_data.columns
        for data in pd_data.values:
            prob_dict = {}
            prob = 0
            max_prob = 0
            max_class = None
            for class_v in self._conditional_probability_dict:
                prob = self._prior_probability_dict[class_v]
                for i, attr_v in enumerate(data):
                    if labels[i] in self._continue_attr_set:
                        # print(i, class_v, labels[i])
                        x_mean = self._conditional_probability_dict[class_v][labels[i]]['mean']
                        x_std = self._conditional_probability_dict[class_v][labels[i]]['std']
                        prob *= self._calc_gaussian(attr_v, x_mean, x_std)
                    else:
                        # print(i, class_v, labels[i])
                        prob *= self._conditional_probability_dict[class_v][labels[i]][attr_v]
                prob_dict[class_v] = prob

                # 4.找到最大的 label 输出
                if prob >= max_prob:
                    max_prob = prob
                    max_class = class_v
            print(prob_dict)
            prob_list.append([max_class, max_prob])
        return prob_list

    def _calc_gaussian(self, x_value, x_mean, x_std):
        result = (1 / (np.sqrt(2 * np.pi) * x_std)) * np.e ** ((x_value - x_mean) ** 2 / (-2 * x_std ** 2))
        # print(result)
        return result


def test_and_draw(pd_data, k=4, figsize=(16, 8)):
    """
    K折交叉测试 naive bayes 算法分类准确率并画图展示
    :param pd_data: pandas格式的训练数据，最后一列为类别标签
    :param k: 将数据集分为几份
    :param figsize: 画布大小
    :return:
    """
    fig = plt.figure(figsize=figsize)
    kf = KFold(n_splits=k, shuffle=True)
    for i, data_index in enumerate(kf.split(pd_data)):
        train_data_index, test_data_index = data_index
        classifer = MyNaiveBayes()
        classifer.train(pd_data.iloc[train_data_index], pd_data.columns[:-1])
        result = classifer.predict(pd_data.iloc[test_data_index, :-1])

        plt.tight_layout(2)
        ax = fig.add_subplot(k, 1, i+1)
        ax.plot([i[0] for i in result], c='blue', alpha=0.7)
        ax.plot([j for j in pd_data.iloc[test_data_index, -1]], c='green', alpha=0.7)
        whether_equal = [i == j for i, j in zip([i[0] for i in result], [j for j in pd_data.iloc[test_data_index, -1]])]
        ax.set_title("Accuracy: %.2f" % (sum(whether_equal) / len(whether_equal)), color='red')
    plt.show()

if __name__ == '__main__':
    # data_path = r"C:\Users\11\PycharmProjects\data\watermelon_3.csv"
    # # data = pd.read_csv(data_path).iloc[:, [1, 2, 3, 4, 5, 6, 9]]
    # data = pd.read_csv(data_path).iloc[:, 1:]
    # columns = list(data.columns[:-1])
    # # print(data)
    # # print("-"*20)
    # classifier = MyNaiveBayes()
    # classifier.train(data, columns)
    # # classifier.train(data, columns, name='Bayes')
    # result = classifier.predict(data.iloc[6:10, :-1])
    # print(result)

    from sklearn.datasets import load_iris

    feature_iris = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    labels_iris = ['setosa', 'versicolor', 'virginica']
    data_iris = load_iris().data
    target_iris = load_iris().target
    pd_data_iris = pd.DataFrame(data=data_iris, columns=feature_iris)
    pd_data_iris['target'] = target_iris
    test_and_draw(pd_data_iris)



