# -*- coding: utf-8 -*-

"""
Created on 1/26/21 8:36 PM
@Author  : Justin Jiang
@Email   : jw_jiang@pku.edu.com
"""

import os
import ahocorasick
import joblib
import jieba
import numpy as np


class EntityExtractor:
    def __init__(self):
        current_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.vocab_path = os.path.join(current_dir, 'data/vocab.txt')
        self.stopwords_path = os.path.join(current_dir, 'data/stop_words.utf8')
        self.word2vec_path = os.path.join(current_dir, 'data/merge_sgns_bigram_char300.txt')

        self.stopwords = [w.strip() for w in open(self.stopwords_path, 'r', encoding='utf8') if w.strip()]

        # 意图分类模型文件
        self.tfidf_path = os.path.join(current_dir, 'model/tfidf_model.m')
        self.nb_path = os.path.join(current_dir, 'model/intent_reg_model.m')  # 朴素贝叶斯模型
        self.tfidf_model = joblib.load(self.tfidf_path)
        self.nb_model = joblib.load(self.nb_path)

        data_dir = os.path.join(current_dir, 'data/')
        self.disease_path = data_dir + 'disease_vocab.txt'  # 疾病
        self.symptom_path = data_dir + 'symptom_vocab.txt'  # 症状
        self.alias_path = data_dir + 'alias_vocab.txt'  # 别称
        self.complication_path = data_dir + 'complications_vocab.txt'  # 并发症

        self.disease_entities = [w.strip() for w in open(self.disease_path, encoding='utf8') if w.strip()]
        self.symptom_entities = [w.strip() for w in open(self.symptom_path, encoding='utf8') if w.strip()]
        self.alias_entities = [w.strip() for w in open(self.alias_path, encoding='utf8') if w.strip()]
        self.complication_entities = [w.strip() for w in open(self.complication_path, encoding='utf8') if w.strip()]

        self.region_words = list(set(self.disease_entities + self.alias_entities + self.symptom_entities))

        # 构造领域actree
        self.disease_tree = self.build_actree(list(set(self.disease_entities)))
        self.alias_tree = self.build_actree(list(set(self.alias_entities)))
        self.symptom_tree = self.build_actree(list(set(self.symptom_entities)))
        self.complication_tree = self.build_actree(list(set(self.complication_entities)))

        self.symptom_qwds = ['什么症状', '哪些症状', '症状有哪些', '症状是什么', '什么表征', '哪些表征', '表征是什么',
                             '什么现象', '哪些现象', '现象有哪些', '症候', '什么表现', '哪些表现', '表现有哪些',
                             '什么行为', '哪些行为', '行为有哪些', '什么状况', '哪些状况', '状况有哪些', '现象是什么',
                             '表现是什么', '行为是什么']  # 询问症状
        self.cureway_qwds = ['药', '药品', '用药', '胶囊', '口服液', '炎片', '吃什么药', '用什么药', '怎么办',
                             '买什么药', '怎么治疗', '如何医治', '怎么医治', '怎么治', '怎么医', '如何治',
                             '医治方式', '疗法', '咋治', '咋办', '咋治', '治疗方法']  # 询问治疗方法
        self.lasttime_qwds = ['周期', '多久', '多长时间', '多少时间', '几天', '几年', '多少天', '多少小时',
                              '几个小时', '多少年', '多久能好', '痊愈', '康复']  # 询问治疗周期
        self.cureprob_qwds = ['多大概率能治好', '多大几率能治好', '治好希望大么', '几率', '几成', '比例',
                              '可能性', '能治', '可治', '可以治', '可以医', '能治好吗', '可以治好吗', '会好吗',
                              '能好吗', '治愈吗']  # 询问治愈率
        self.check_qwds = ['检查什么', '检查项目', '哪些检查', '什么检查', '检查哪些', '项目', '检测什么',
                           '哪些检测', '检测哪些', '化验什么', '哪些化验', '化验哪些', '哪些体检', '怎么查找',
                           '如何查找', '怎么检查', '如何检查', '怎么检测', '如何检测']  # 询问检查项目
        self.belong_qwds = ['属于什么科', '什么科', '科室', '挂什么', '挂哪个', '哪个科', '哪些科']  # 询问科室
        self.disase_qwds = ['什么病', '啥病', '得了什么', '得了哪种', '怎么回事', '咋回事', '回事',
                            '什么情况', '什么问题', '什么毛病', '啥毛病', '哪种病']  # 询问疾病

    def build_actree(self, wordlist):
        """
        构造actree，加速过滤

        :param wordlist:
        :return:
        """
        actree = ahocorasick.Automaton()
        # 向树中添加新单词
        for idx, word in enumerate(wordlist):
            actree.add_word(word, (idx, word))
        actree.make_automaton()
        return actree

    def entity_reg(self, question):
        """
        模式匹配, 得到匹配的词和类型。如疾病，疾病别名，并发症，症状
        :param question:str
        :return:
        """
        self.result = {}
        # 抽取问题中的疾病关键词
        for idx in self.disease_tree.iter(question):
            word = idx[1][1]  # 取出命中的关键词，idx的格式为(命中关键词在输入的结尾位置，（关键词的索引，关键词）)
            if "Disease" not in self.result:  # 若结果中尚未出现疾病类关键词，则加入该栏位
                self.result['Disease'] = [word]
            else:
                self.result['Disease'].append(word)

        # 抽取问题中的别称关键词
        for idx in self.alias_tree.iter(question):
            word = idx[1][1]
            if "Alias" not in self.result:
                self.result["Alias"] = [word]
            else:
                self.result["Alias"].append(word)

        # 抽取问题中的症状关键词
        for i in self.symptom_tree.iter(question):
            wd = i[1][1]
            if "Symptom" not in self.result:
                self.result["Symptom"] = [wd]
            else:
                self.result["Symptom"].append(wd)

        # 抽取问题中的并发症关键词
        for i in self.complication_tree.iter(question):
            wd = i[1][1]
            if "Complication" not in self.result:
                self.result["Complication"] = [wd]
            else:
                self.result["Complication"].append(wd)

        return self.result

    def find_sim_words(self, question):
        """
        当全匹配失败时，就采用相似度计算来找相似的词
        :param question:
        :return:
        """
        import re
        import string
        from gensim.models import KeyedVectors

        jieba.load_userdict(self.vocab_path)
        print("初步匹配失败，触发模型相似度匹配模块，模型加载中请稍后...")
        self.model = KeyedVectors.load_word2vec_format(self.word2vec_path, binary=False)
        print("word2vec model loaded...")
        # 去除符号、空格
        sentence = re.sub("[{}]", re.escape(string.punctuation), question)
        sentence = re.sub("[，。‘’；：？、！【】]", " ", sentence)
        sentence = sentence.strip()

        # 切词，去停用词与长度小于2的词
        words = [w.strip() for w in jieba.cut(sentence) if w.strip() not in self.stopwords and len(w.strip()) >= 2]

        alist = []

        for word in words:
            temp = [self.disease_entities, self.alias_entities, self.symptom_entities, self.complication_entities]
            for i in range(len(temp)):
                flag = ''
                if i == 0:
                    flag = "Disease"
                elif i == 1:
                    flag = "Alias"
                elif i == 2:
                    flag = "Symptom"
                else:
                    flag = "Complication"
                scores = self.simCal(word, temp[i], flag)
                alist.extend(scores)
        temp1 = sorted(alist, key=lambda k: k[1], reverse=True)
        if temp1:
            self.result[temp1[0][2]] = [temp1[0][0]]

    def editDistanceDP(self, s1, s2):
        """
        采用DP方法计算编辑距离
        :param s1:
        :param s2:
        :return:
        """
        m = len(s1)
        n = len(s2)
        solution = [[0 for j in range(n + 1)] for i in range(m + 1)]
        for i in range(len(s2) + 1):
            solution[0][i] = i
        for i in range(len(s1) + 1):
            solution[i][0] = i

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    solution[i][j] = solution[i - 1][j - 1]
                else:
                    solution[i][j] = 1 + min(solution[i][j - 1], min(solution[i - 1][j],
                                                                     solution[i - 1][j - 1]))
        return solution[m][n]

    def simCal(self, word, entities, flag):
        """
        计算词语和字典中的词的相似度
        相同字符的个数/min(|A|,|B|)   +  余弦相似度
        :param word: str
        :param entities:List
        :return:
        """
        a = len(word)
        scores = []
        for entity in entities:
            sim_num = 0
            b = len(entity)
            c = len(set(entity + word))
            temp = []
            for w in word:
                if w in entity:
                    sim_num += 1
            if sim_num != 0:
                score1 = sim_num / c  # overlap score
                temp.append(score1)
            try:
                score2 = self.model.similarity(word, entity)  # 余弦相似度分数
                temp.append(score2)
            except:
                pass
            score3 = 1 - self.editDistanceDP(word, entity) / (a + b)  # 编辑距离分数
            if score3:
                temp.append(score3)

            score = sum(temp) / len(temp)
            if score >= 0.7:
                scores.append((entity, score, flag))

        scores.sort(key=lambda k: k[1], reverse=True)
        return scores

    def check_words(self, wds, sent):
        """
        基于特征词分类,若wds中的wd在sent中则返回True
        :param wds:
        :param sent:
        :return:
        """
        for wd in wds:
            if wd in sent:
                return True
        return False

    def tfidf_features(self, text, vectorizer):
        """
        提取问题的TF-IDF特征
        :param text:
        :param vectorizer:
        :return:
        """
        jieba.load_userdict(self.vocab_path)
        words = [w.strip() for w in jieba.cut(text) if w.strip() and w.strip() not in self.stopwords]
        sents = [' '.join(words)]

        tfidf = vectorizer.transform(sents).toarray()
        return tfidf

    def other_features(self, text):
        """
        提取问题的关键词特征，计算text中的询问关键词频次以确定询问重点
        :param text:
        :return:
        """
        features = [0] * 7
        for d in self.disase_qwds:
            if d in text:
                features[0] += 1

        for s in self.symptom_qwds:
            if s in text:
                features[1] += 1

        for c in self.cureway_qwds:
            if c in text:
                features[2] += 1

        for c in self.check_qwds:
            if c in text:
                features[3] += 1
        for p in self.lasttime_qwds:
            if p in text:
                features[4] += 1

        for r in self.cureprob_qwds:
            if r in text:
                features[5] += 1

        for d in self.belong_qwds:
            if d in text:
                features[6] += 1

        # 计算权重
        m = max(features)
        n = min(features)
        normed_features = []
        if m == n:
            normed_features = features
        else:
            for i in features:
                j = (i - n) / (m - n)
                normed_features.append(j)

        return np.array(normed_features)

    def model_predict(self, x, model):
        """
        预测意图
        :param x:
        :param model:
        :return:
        """
        pred = model.predict(x)
        return pred

    # 实体抽取主函数
    def extractor(self, question):
        self.entity_reg(question)  # 进行第一次关键词匹配，使用ACTree逐字匹配
        if not self.result:
            self.find_sim_words(question)  # 若匹配不到任何东西，则计算相似词，涉及余弦相似度与编辑距离

        types = []  # 实体类型
        for v in self.result.keys():
            types.append(v)  # 将已经识别出实体的实体类型加入至types中

        intentions = []  # 查询意图

        tfidf_feature = self.tfidf_features(question, self.tfidf_model)  # 计算输入的tfidf特征(1,830)
        other_feature = self.other_features(question)  # 计算询问关键词频次特征(7,)
        m = other_feature.shape
        other_feature = np.reshape(other_feature, (1, m[0]))  # 将一维序列转为二维矩阵,(1,7)

        feature = np.concatenate((tfidf_feature, other_feature), axis=1)  # 将两个特征的二维矩阵合并，形成一(1,837)矩阵

        predicted = self.model_predict(feature, self.nb_model)  # 取得询问意图结果，如['query_department']
        intentions.append(predicted[0])  # 添加询问意图类型至intentions

        # 预设询问句式，添加打中的意图至intentions中
        # 已知疾病，查询症状
        if self.check_words(self.symptom_qwds, question) and ('Disease' in types or 'Alia' in types):
            intention = "query_symptom"
            if intention not in intentions:
                intentions.append(intention)
        # 已知疾病或症状，查询治疗方法
        if self.check_words(self.cureway_qwds, question) and \
                ('Disease' in types or 'Symptom' in types or 'Alias' in types or 'Complication' in types):
            intention = "query_cureway"
            if intention not in intentions:
                intentions.append(intention)
        # 已知疾病或症状，查询治疗周期
        if self.check_words(self.lasttime_qwds, question) and ('Disease' in types or 'Alia' in types):
            intention = "query_period"
            if intention not in intentions:
                intentions.append(intention)
        # 已知疾病，查询治愈率
        if self.check_words(self.cureprob_qwds, question) and ('Disease' in types or 'Alias' in types):
            intention = "query_rate"
            if intention not in intentions:
                intentions.append(intention)
        # 已知疾病，查询检查项目
        if self.check_words(self.check_qwds, question) and ('Disease' in types or 'Alias' in types):
            intention = "query_checklist"
            if intention not in intentions:
                intentions.append(intention)
        # 查询科室
        if self.check_words(self.belong_qwds, question) and \
                ('Disease' in types or 'Symptom' in types or 'Alias' in types or 'Complication' in types):
            intention = "query_department"
            if intention not in intentions:
                intentions.append(intention)
        # 已知症状，查询疾病
        if self.check_words(self.disase_qwds, question) and ("Symptom" in types or "Complication" in types):
            intention = "query_disease"
            if intention not in intentions:
                intentions.append(intention)

        # 若没有检测到意图，且已知疾病，则返回疾病的描述
        if not intentions and ('Disease' in types or 'Alias' in types):
            intention = "disease_describe"
            if intention not in intentions:
                intentions.append(intention)
        # 若是疾病和症状同时出现，且出现了查询疾病的特征词，则意图为查询疾病
        if self.check_words(self.disase_qwds, question) and ('Disease' in types or 'Alias' in types) \
                and ("Symptom" in types or "Complication" in types):
            intention = "query_disease"
            if intention not in intentions:
                intentions.append(intention)
        # 若没有识别出实体或意图则调用其它方法
        if not intentions or not types:
            intention = "QA_matching"
            if intention not in intentions:
                intentions.append(intention)

        self.result["intentions"] = intentions  # 最终总结意图
        return self.result


if __name__ == '__main__':
    test = EntityExtractor()

    question = "产后三急属于什么科啊 不帮我"
    print(test.extractor(question))
