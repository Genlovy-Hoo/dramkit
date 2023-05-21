
# TODO:
# 由两列生成树
# 由树生成两列
# 给定节点查找父树和子树
# 给定节点查找所有父或子节点列表

if __name__ == '__main__':
    ls=[("a","a1"),
        ("a","a2"),
        ("a","a3"),
        ("a1","b1"),
        ("a2","b2"),
        ("a2","b3"),
        ]
     
    def get_node_childrens(parent,ls):
        nodes=[]
        for row in ls:
            if row[0]==parent["id"]:
                node={"id":row[1],"children":[]}
                nodes.append(node)
        return nodes
    def node_tree(parent,ls):
        nodes=get_node_childrens(parent,ls)
        if len(nodes)>0:
            for node in nodes:
                parent["children"].append(node)
            for node in parent["children"]:
                node_tree(node,ls)
    def get_node_tail(tree,tail):
        ch=tree["children"]
        if ch:
            for node in ch:
                get_node_tail(node,tail)
        else:
            tail.append(tree["id"])
     
    tree={"id":"a","children":[]}
    node_tree(tree,ls)
    print(tree)
    tail=[]
    get_node_tail(tree,tail)
    print(tail)