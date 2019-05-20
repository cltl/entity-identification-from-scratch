import networkx as nx

filename='bin/mention_docid_graph.graph'
g=nx.read_gpickle(filename)
a_node='http://cltl.nl/entity#Franse2036'
print(len(g.adj[a_node]))
