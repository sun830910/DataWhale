# -*- coding: utf-8 -*-

"""
Created on 3/16/21 4:03 PM
@Author  : Justin Jiang
@Email   : jw_jiang@pku.edu.com
"""


from collections import namedtuple

# 使用具名元组定义特征标记
SparseFeat = namedtuple('SparseFeat', ['name', 'vocabulary_size', 'embedding_dim'])
DenseFeat = namedtuple('DenseFeat', ['name', 'dimension'])
VarLenSparseFeat = namedtuple('VarLenSparseFeat', ['name', 'vocabulary_size', 'embedding_dim', 'maxlen'])
