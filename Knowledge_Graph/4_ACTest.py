# -*- coding: utf-8 -*-

"""
Created on 1/27/21 3:33 PM
@Author  : Justin Jiang
@Email   : jw_jiang@pku.edu.com

ahocorasick实现快速的关键字匹配
"""

import ahocorasick

if __name__ == '__main__':
    A = ahocorasick.Automaton()

    # 建立Pattern tree；即建立自动机，简单来说就是根据输入的字符串构造一棵“树”；
    for idx, key in enumerate('我 经常 在 微信 公众号 码龙社 看 文章'.split()):
        A.add_word(key, (idx, key))

    print('文章' in A)
    print('测试' in A)

    print(A.get('我'))

    A.make_automaton()

    sentence = '我 在 这'
    for end_index, (insert_order, original_value) in A.iter(sentence):
        start_index = end_index - len(original_value) + 1
        print((start_index, end_index, (insert_order, original_value)))
        assert sentence[start_index:start_index + len(original_value)] == original_value

    print('-' * 30)
    sentence = '我 爱 北京 天安门'
    for end_index, (insert_order, original_value) in A.iter(sentence):
        start_index = end_index - len(original_value) + 1
        print((start_index, end_index, (insert_order, original_value)))
        assert sentence[start_index:start_index + len(original_value)] == original_value
