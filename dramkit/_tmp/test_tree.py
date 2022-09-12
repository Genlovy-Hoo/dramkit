# -*- coding: utf-8 -*-

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from dramkit.gentools import isnull
from dramkit.iotools import load_csv
    

def tree_from_edgs(edgs):
    graph = nx.Graph()
    for n1, n2 in edgs:
        graph.add_edge(n1, n2)
    return graph


def tree_from_node_and_parent(nodes):
    tree = [node for node in nodes if node['parent_id'] is None]
    for node in nodes:
        node['children'] = [n for n in nodes if node['id'] == n['parent_id']]
    return tree

    
def get_tree_from_node_and_children(df):
    '''根据df中的node列和children_node列生成树状关系'''
    parents = df['node'].unique().tolist()
    childs = df['children_node'].unique().tolist()
    roots = [x for x in parents if x not in childs]


if __name__ == '__main__':
    
    # get_tree_from_node_and_children
    df = load_csv('./model_relation.csv')
    
    
    
    relations = [('a', 'b'), ('a', 'c'), ('a', 'd'), ('c', 'f'),
                 ('b', 'e'), ('d', 'g'), ('d', 'h'), ('d', 'i'),
                 ('e', 'j'), ('f', 'k'), ('g', 'l'), ('h', 'n'),
                 ('g', 'm'), ('h', 'o'), ('i', 'p'), ('i', 'q'),
                 ('x', 'y'), ('x', 'z')]
    
    df_ = df.dropna(subset=['children_node'])
    relations = []
    for k in range(df_.shape[0]):
        relations.append((df_['node'].iloc[k],
                          df_['children_node'].iloc[k]))
    
    # tree = tree_from_edgs(relations)
    # nx.draw(tree, with_labels=True)
    # plt.savefig('test.png')
    
    
    
    
    nodes = [
        {'id': 1, 'name': '1', 'parent_id': None},
        {'id': 2, 'name': '2', 'parent_id': 1},
        {'id': 3, 'name': '3', 'parent_id': 1},
        {'id': 4, 'name': '4', 'parent_id': 2},
        {'id': 5, 'name': '5', 'parent_id': 3},
        {'id': 6, 'name': '6', 'parent_id': 5},
    ]
    tree1 = tree_from_node_and_parent(nodes)
    
    
    parents = df['node'].dropna().unique().tolist()
    childs = df['children_node'].dropna().unique().tolist()
    roots = list(set([x for x in parents if x not in childs]))
    # names = list(set(parents + childs))
    # names_id = {names[k]: k for k in range(len(names))}
    
    # nodes = []
    # for rt in roots:
    #     nodes.append({'id': names_id[rt],
    #                   'name': rt,
    #                   'parent_id': None})
    # for k in range(df.shape[0]):
    #     if isnull(df['children_node'].iloc[k]):
    #         continue
    #     else:
    #         child = df['children_node'].iloc[k]
    #         parent = df['node'].iloc[k]
    #         nodes.append({'id': names_id[child],
    #                       'name': child,
    #                       'parent_id': names_id[parent]})
    # tree2 = tree_from_node_and_parent(nodes)
    
    base = []
    for k in range(df.shape[0]):
        if pd.isna(df['children_node'].iloc[k]):
            base.append(df['node'].iloc[k])














