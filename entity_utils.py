import classes

import networkx as nx
from pprint import pprint
import pickle
import spacy
import sys
import itertools
import numpy as np
from collections import Counter, defaultdict
from sklearn.metrics.pairwise import cosine_similarity

import nl_core_news_sm
nl_nlp=nl_core_news_sm.load()

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

def compute_similarity(w1, w2, vectors):
    """Compute similarity of 2 vectors."""
    v1=np.array(vectors[w1]).reshape(1, -1)
    v2=np.array(vectors[w2]).reshape(1, -1)
    return cosine_similarity(v1, v2)

def load_sentences(data):
    """Given a set of classes with entities and links, generate embeddings."""

    all_sentences=[]
    for news_item in data:
        new_content=replace_entities(str(news_item.content), news_item.sys_entity_mentions)
        text=f"{news_item.title}\n{new_content}"
        nl_doc=nl_nlp(text)
        for sent in nl_doc.sents:
            sent_tokens = [t.text for t in sent]
            all_sentences.append(sent_tokens)
    return all_sentences

def replace_entities(text, mentions):
    to_replace={}
    for e in mentions:
        to_replace[e.begin_index]=e.identity.lstrip('http://cltl.nl/entity#').replace(' ', '_')
        for i in range(e.begin_index+1, e.end_index):
            to_replace[i]=''
    doc=nl_nlp(text)
    new_text=[]
    for t in doc:
        idx=t.i
        token=t.text
        if idx in to_replace:
            if to_replace[idx]:
                new_text.append(to_replace[idx])
        else:
            new_text.append(token)
    if 'Couliba' in to_replace.values() or 'Coulibaly' in to_replace.values():
        print(new_text)
    return ' '.join(new_text)

def generate_graph(data, filename):
    """
    Generate undirected graph, given a collection of news documents.
    """
    G=nx.Graph()
    for news_item in data:
        for mention in news_item.sys_entity_mentions:
            identity=mention.identity
            G.add_node(identity)
            for other_mention in news_item.sys_entity_mentions:
                other_identity=other_mention.identity
                if other_identity>identity:
                    G.add_edge(identity, other_identity)
    print(G.number_of_nodes())
    print(G.number_of_edges())
    
    nx.write_gpickle(G, filename.replace('.pkl', '.graph'))

def get_variable_len_combinations(arr):
    """Get combinations of factors with length 0 to len(arr)"""
    res = []
    for l in range(0, len(arr)+1):
        for x in itertools.combinations(arr, l):
            res.append(x)
    return res

def recognize_entities(news_items):
    """
    Run NER on all documents.
    """
    import nl_core_news_sm
    nl_nlp=nl_core_news_sm.load()
    for i, news_item in enumerate(news_items):
        text=f"{news_item.title}\n{news_item.content}"
        nl_doc=nl_nlp(text)
        for e in nl_doc.ents:
            ent_mention_obj=classes.EntityMention(
                mention=e.text,
                begin_index=e.start,
                end_index=e.end,
                the_type=e.label_
            )
            news_item.sys_entity_mentions.append(ent_mention_obj)
        print(i)
    return news_items
