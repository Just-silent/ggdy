# coding:UTF-8
# author    :Just_silent
# init time :2021/5/13 14:57
# file      :response.py
# IDE       :PyCharm

import ahocorasick
import pandas as pd

from py2neo import Graph
from train.similar import SIMI, SimCSE


class Response():
    '''
    通过nlp分析和neo4j的查询返回结果，主要包括：关键词抽取，问题锁定
    '''
    def __init__(self):
        self.simi = SIMI()
        self.simCSE = SimCSE()
        self.simi.load_model()
        self.simCSE.load_model()
        self.graf = Graph(
            "http://172.22.179.237:7474/",
            user="neo4j",
            password="123456"
        )
        self.wordlist = []
        pd_data1 = pd.read_excel(r"E:\Github\inemail\data\邮箱用户手册整理.xlsx", header=0, sheet_name='neo')
        self.wordlist = list(set(pd_data1['keyword'].values))
        self.actree = ahocorasick.Automaton()
        for index, word in enumerate(self.wordlist):
            self.actree.add_word(word, (index, word))
        self.actree.make_automaton()
        pass

    def get_keyword(self, sentence):
        '''
        两种方式获取最终的keyword：1.直接匹配 2.匹配结果映射（a:一个词向另一个词映射 b:多个分词向一个词映射）
        策略：
            1. 如果多个词有重叠部分，只保留最长的那个keyword

        :param sentence: 句子
        :return: keyword
        '''
        keywords = []
        for item in self.actree.iter_long(sentence):  # 将AC_KEY中的每一项与content内容作对比，若匹配则返回
            keywords.append(item[1][1])
        new_keywords = list(set(keywords))
        new_keywords.sort(key=keywords.index)
        return keywords

    def get_all_keys(self):
        all_keys = []
        sql = 'MATCH (n:keyword) RETURN n'
        x = self.graf.run(sql).data()
        for node in x:
            all_keys.append(node['n']['name'])
        return all_keys

    def get_rel_questions(self, keyword):
        # 获取和关键词相关的问题
        response = {}
        sql_question = 'match (:keyword{name:\'' + keyword + '\'})-[:`' + 'keyword_question' + '`]->(question)-[:`' + 'question_operate' + '`]->(operate) return question'
        sql_operate = 'match (:keyword{name:\'' + keyword + '\'})-[:`' + 'keyword_question' + '`]->(question)-[:`' + 'question_operate' + '`]->(operate) return operate'
        qs = self.graf.run(sql_question).data()
        os = self.graf.run(sql_operate).data()
        for q, o in zip(qs, os):
            response[q['question']['name']] = o['operate']['name']
        return response

    def get_max_p_response(self, input, questions_operates):
        # 计算输入与相关问题的相似度，并匹配最大概率的问题与答案
        questions = list(questions_operates.keys())
        index, p = self.simi.predict_many(input, questions)
        index1, p1 = self.simCSE.simCSE_predict_many(input, questions)
        print(index, p)
        response = questions_operates[questions[index]]
        response1 = questions_operates[questions[index1]]
        return questions[index], response, round(p*100, 2), questions[index1], response1, round(p1*100, 2)

    def get_recommend_question(self, keywords):
        # 获取n个关键词有关的所有问题及其答案
        questions = []
        for keyword in keywords:
            questions_operates = self.get_rel_questions(keyword)
            questions.extend(list(questions_operates.keys()))
        questions_rm = list(set(questions))
        questions_rm.sort(key=questions.index)
        response = '您要问的事下面的那个问题：<br/>'
        for i in range(len(questions_rm)):
            response += '{} : {}<br/>'.format(i+1, questions_rm[i])
        response += '你可以输入123代表问题！'
        return response, questions_operates
        pass

    def _init_history(self, history):
        history['is_right'] = None
        history['num_keys'] = None
        history['question_operate'] = None
        history['keys'] = None
        history['response'] = None
        history['is_asking'] = 0
        history['question'] = None
        history['last_step'] = None

    def get_response(self, input, history):
        # 回答是否正确 or 结束本轮对话
        # 根据用户回答，回答正确 或 转人工，退出多轮对话
        if input.isdigit() and int(input)==10000:
            self._init_history(history)
            return '很高兴展开新一轮对话！'
        else:
            # 第一次对话， 获取keys和num_keys
            if history['num_keys']==None:
                history['question'] = input
                keys = self.get_keyword(input)
                history['num_keys'] = len(keys)
                if len(keys)!=0:
                    history['keys'] = keys
                return self.get_response(input, history)
            # 无法抽取关键词
            if history['num_keys']==0:
                # 推荐关键词
                if history['keys']==None and (input.isdigit() is False):
                    history['keys'] = self.get_all_keys()
                    response = '您要问的是否关于如下方面：<br/>'
                    for i, key in enumerate(history['keys']):
                        response+='{}: {}<br/>'.format(i+1, key)
                    return response
                elif history['keys']==None and input.isdigit() and (int(input)>len(history['keys']) or int(input)<=0):
                    return '您的输入有误，请继续上面的回答，请重新输入对应数字！<br/>' \
                            '若想开启新的对话，请输入10000！'
                elif history['keys']!=None and input.isdigit():
                    key = history['keys'][int(input)-1]
                    history['keys'] = [key]
                    history['num_keys'] = 1
                    return self.get_response(input, history)
                # 用户提供关键词
            # 抽取一个关键词
            elif history['num_keys'] == 1:
                if history['question_operate']==None:
                    # 预测输入与所对应的问题(需要询问)
                    key = history['keys'][0]
                    questions_operates = self.get_rel_questions(key)
                    print(history['question'])
                    print(questions_operates)
                    question, response, p, question1, response1, p1 = self.get_max_p_response(history['question'], questions_operates)
                    response =  '有监督：<br/>' \
                                '最佳问题匹配：{}<br/>'\
                                '概率：{}%<br/>' \
                                '结果：{}<br/>'.format(question, p, response)
                    response += '无监督：<br/>' \
                                '最佳问题匹配：{}<br/>' \
                               '概率：{}%<br/>' \
                               '结果：{}<br/>'.format(question1, p1, response1)
                    response += '龙龙bot的回答是否正确？<br/>' \
                                '0：错误；1：正确'
                    history['response'] = response # 需要询问
                    history['is_asking'] = 1 # 询问回答是否正确
                    history['question_operate'] = questions_operates
                    return response
                elif history['is_asking']==1:
                    if input.isdigit() and (int(input)==0 or int(input)==1):
                        history['is_right'] = True if int(input)==1 else False
                        history['is_asking'] = -1
                        return self.get_response(history['question'], history)
                    else:
                        return '您的输入有误，请继续上面的回答，请重新输入0或1！<br/>' \
                               '若想开启新的对话，请输入10000！'
                # 匹配的问题错误，推荐所有问题
                elif history['is_right']== False:
                    if input.isdigit()==False:
                        response = '您要问的是否如下问题：<br/>'
                        for i, key in enumerate(list(history['question_operate'].keys())):
                            response += '{}: {}<br/>'.format(i + 1, key)
                        response += '<br/>0: 表示没有您要的问题！'
                        return response
                    elif input.isdigit() and int(input)<=len(history['question_operate'].keys()) and int(input)>0:
                        x = list(history['question_operate'].keys())[int(input)-1]
                        response = '最佳问题匹配：{}<br/>'\
                                   '结果：{}<br/>' \
                                   '期待您的再次咨询😄'.format(x, history['question_operate'][x])
                        self._init_history(history)
                        return response
                    elif input.isdigit() and int(input)==0:
                        self._init_history(history)
                        return '好的，马上给您连接人工服务！'
                elif history['is_right'] == True:
                    self._init_history(history)
                    print('完成回答，初始化并退出', history)
                    return '能正确回答您的问题，龙龙义不容辞😄'
                pass
            # 抽取多个关键词
            elif history['num_keys'] > 1:
                # 推荐关键词
                if input.isdigit() is False:
                    response = '您要问的是否关于如下方面：<br/>'
                    for i, key in enumerate(history['keys']):
                        response += '{}: {}<br/>'.format(i + 1, key)
                    return response
                elif input.isdigit() and (int(input) > len(history['keys']) or int(input) <= 0):
                    return '没有你输入的关键词，请重新输入对应的序号！'
                elif input.isdigit():
                    key = history['keys'][int(input) - 1]
                    history['keys'] = [key]
                    history['num_keys'] = 1
                    return self.get_response(input, history)
        pass

    def evaluator(self, input):
        keywords = self.get_keyword(input)
        question = ''
        p=5
        if len(keywords) == 1:
            questions_operates = self.get_rel_questions(keywords[0])
            question, response, p, question1, response1, p1 = self.get_max_p_response(input, questions_operates)
        return question, p


if __name__ == '__main__':
    history = {
        'is_right': None,
        'num_keys': None,
        'question_operate': None,
        'keys': None,
        'response': None,
        'is_asking': 0,
        'question': None
    }
    response = Response()
    input = '我爱中国'
    response = response.get_response(input, history)
    print(response)