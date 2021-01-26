# -*- coding: utf-8 -*-

"""
Created on 1/26/21 11:28 AM
@Author  : Justin Jiang
@Email   : jw_jiang@pku.edu.com
"""

from neo4j import GraphDatabase

# 连接 Neo4j 图数据库

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "1234"))


def add_friend(tx, name, friend_name):
    tx.run("MERGE (a:Person {name: $name}) "
           "MERGE (a)-[:KNOWS]->(friend:Person {name: $friend_name})",
           name=name, friend_name=friend_name)


# 定义 关系函数
def print_friends(tx, name):
    for record in tx.run("MATCH (a:Person)-[:KNOWS]->(friend) WHERE a.name = $name "
                         "RETURN friend.name ORDER BY friend.name", name=name):
        print(record["friend.name"])


# step 3：运行
with driver.session() as session:
    session.write_transaction(add_friend, "Arthur", "Guinevere")
    session.write_transaction(add_friend, "Arthur", "Lancelot")
    session.write_transaction(add_friend, "Arthur", "Merlin")
    session.read_transaction(print_friends, "Arthur")
