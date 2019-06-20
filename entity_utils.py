import classes

import networkx as nx
from pprint import pprint
import pickle
import spacy
import sys
import itertools
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity(w1, w2, vectors):
    """Compute similarity of 2 vectors."""
    v1=np.array(vectors[w1]).reshape(1, -1)
    v2=np.array(vectors[w2]).reshape(1, -1)
    return cosine_similarity(v1, v2)

def load_sentences(data):
    """Given a set of classes with entities and links, generate embeddings."""
    import nl_core_news_sm
    nl_nlp=nl_core_news_sm.load()
    
    all_sentences=[]
    for news_item in data:
        text=f"{news_item.title}\n{news_item.content}"
        nl_doc=nl_nlp(text)
        for sent in nl_doc.sents:
            sent_tokens = [t.text for t in sent]
            all_sentences.append(sent_tokens)
    return all_sentences

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