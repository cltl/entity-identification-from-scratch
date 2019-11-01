import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, defaultdict
import numpy as np

def inspect(data, with_types=False, graph=None):
    num_mentions=0
    identities=set()
    occurence_types=[]
    instance_types={}
    degrees_per_type=defaultdict(list)
    
    max_degree=0
    max_degree_node=None
    
    print(len(data), 'news documents')
    for news_item_obj in data:
        for m in news_item_obj.sys_entity_mentions:
            identities.add(m.identity)
            if with_types:
                a_type=m.the_type
                occurence_types.append(a_type)
                instance_types[m.identity]=a_type
        num_mentions+=len(news_item_obj.sys_entity_mentions)

    print('Num mentions', num_mentions)
    print('Num identities', len(identities))
    
    if with_types:
        print('Type distribution of occurrences', Counter(occurence_types))
        print('Type distribution of aggregated instances', Counter(instance_types.values()))
    
    if graph:
        for identity in instance_types.keys():
            degree=len(graph.adj[identity])
            degrees_per_type[identity[-3:]].append(degree)
            if degree>max_degree:
                max_degree=degree
                max_degree_node=identity

        for k,v in degrees_per_type.items():
            print(k, round(np.mean(v),1), '/', round(np.std(v),1))
        print('Max degree node', max_degree_node, max_degree)
        
    return identities


def inspect_data(data, graph_file):
    """Analyze basic properties of the data."""
    with_types='type' in graph_file
    g=None
    if with_types:
        g=nx.read_gpickle(graph_file)
    ids=inspect(data, with_types, g)
    return ids

def compute_similarity(w1, w2, vectors):
    """Compute similarity of 2 vectors."""
    try:
        v1=np.array(vectors[w1]).reshape(1, -1)
    except KeyError:
        print(w1, 'not in vocab')
        return 0
    try:
        v2=np.array(vectors[w2]).reshape(1, -1)
    except KeyError:
        print(w2, 'not in vocab')
        return 0
    return cosine_similarity(v1, v2)
