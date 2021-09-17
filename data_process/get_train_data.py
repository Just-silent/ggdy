# coding:UTF-8
# author    :ZHX、Just_silent
# init time :2021/5/12 15:22
# file      :get_simi_data.py
# IDE       :PyCharm

import random
import pandas as pd

from train.path import *


class SimiData():
    '''处理原始数据->训练数据
    '''
    def __init__(self, xlsx_path, train_data_path):
        '''
        :param xlsx_path: 原始数据
        :param train_data_path: 训练数据
        '''
        self.all_question = {}
        self.all_q = []
        self.pd_data = pd.read_excel(xlsx_path, sheet_name='neo', header=0)
        self.file_w = open(train_data_path, 'w', encoding='utf-8')
        self._data2list()
        pass

    def _data2list(self):
        # 提取xlsx中的元素，并整理成list的形式
        for index, item in self.pd_data.iterrows():
            keyword = item['keyword']
            question = item['question']
            another = item['another'].split('、')
            questions = []
            self.all_q.append(question)
            if item['keyword'] not in self.all_question.keys():
                self.all_question[keyword] = []
            questions.append(question)
            for que in another:
                questions.append(que)
            self.all_question[keyword].append(questions)
        pass

    def _write_simi_data(self):
        '''提取xlsx中的数据，并整理整成 (s1, s2, 1)的格式
        '''
        for key, item in self.all_question.items():
            for i in range(0, len(item)):
                # 相关
                for ii in range(1, len(item[i])):
                    self.file_w.write(item[i][0] + '\t' + item[i][ii] + '\t' + str(1) + '\n')
                # 不相关1
                for j in range(i + 1, len(item)):
                    for jj in range(0, len(item[j])):
                        self.file_w.write(item[i][0] + '\t' + item[j][jj] + '\t' + str(0) + '\n')
        pass

    def _write_no_simi_data(self):
        '''提取xlsx中的数据，并整理整成 (s1, s2, 0)的格式
        '''
        for index, que in enumerate(self.all_q):
            another_indexs = random.sample(range(0, len(self.all_q)), 4)
            if index in another_indexs:
                another_indexs.remove(index)
            for ano in another_indexs:
                self.file_w.write(self.all_q[index] + '\t' + self.all_q[ano] + '\t' + str(0) + '\n')
        self.file_w.close()
        pass

    def get_train_data(self):
        self._write_simi_data()
        self._write_no_simi_data()
        print('训练数据处理完毕！')

if __name__ == '__main__':
    xlsx_path = xlsx_path
    train_data_path = train_data_path
    simi_data = SimiData(xlsx_path, train_data_path)
    simi_data.get_train_data()