# -*- coding: utf-8 -*-

"""
Created on 1/27/21 5:48 PM
@Author  : Justin Jiang
@Email   : jw_jiang@pku.edu.com
"""

from entity_extractor import EntityExtractor
from search_answer import AnswerSearching


class KBQA:
    def __init__(self):
        self.extractor = EntityExtractor()
        self.searcher = AnswerSearching()
        self.answer = "对不起，您的问题我不知道，我今后会努力改进的。"

    def qa_query(self, question):
        entities = self.extractor.extractor(question)
        if not entities:  # 没catch到任何实体
            return self.answer
        sqls = self.searcher.question_parser(entities)
        final_answer = self.searcher.searching(sqls)
        if not final_answer:
            return self.answer
        else:
            return '\n'.join(final_answer)


if __name__ == '__main__':
    handler = KBQA()
    print("欢迎使用基于医疗知识图谱的问答系统")
    while True:
        question = input("请输入您的问题:")
        if not question:
            print("服务结束，谢谢您的使用")
            break
        answer = handler.qa_query(question)
        print("查询结果为：", answer)
        print("-" * 30)
