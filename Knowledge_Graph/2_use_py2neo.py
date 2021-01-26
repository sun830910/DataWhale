# -*- coding: utf-8 -*-

"""
Created on 1/26/21 2:47 PM
@Author  : Justin Jiang
@Email   : jw_jiang@pku.edu.com
"""

# step 1：导包
from py2neo import Graph, Node, Relationship

# step 2：构建图
g = Graph()
# step 3：创建节点
tx = g.begin()
a = Node("Person", name="Alice")
tx.create(a)
b = Node("Person", name="Bob")
# step 4：创建边
ab = Relationship(a, "KNOWS", b)
# step 5：运行
tx.create(ab)
tx.commit()