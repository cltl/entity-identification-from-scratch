import classes

import networkx as nx
import copy
import os.path
from pprint import pprint
import glob
import json
import pickle
import spacy

def load_news_items(loc):
    """
    Load news items into objects defined in the classes file.
    """
    
    news_items=set()
    for file in glob.glob('%s/*.json' % loc):
        with open(file, 'r') as f:
            data=json.load(f)
        news_item_obj=classes.NewsItem(
            content=data['body'],
            title=data['title'],
            identifier=file.split('.')[0],
            collection=loc
        )
        news_items.add(news_item_obj)
    return news_items

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
            news_item.entity_mentions.append(ent_mention_obj)
        print(i)
    return news_items

def generate_graph(data, filename):
    """
    Generate undirected graph, given a collection of news documents.
    """
    G=nx.Graph()
    for news_item in data:
        for mention in news_item.entity_mentions:
            identity=mention.identity
            G.add_node(identity)
            for other_mention in news_item.entity_mentions:
                other_identity=other_mention.identity
                if other_identity>identity:
                    G.add_edge(identity, other_identity)
    print(G.number_of_nodes())
    print(G.number_of_edges())
    
    nx.write_gpickle(G, filename.replace('.pkl', '.graph'))

def generate_identity(objs,prefix='http://cltl.nl/entity#', factors=[]):
    """
    Decide which entities are identical, based on a set of recognized entity mentions and flexible set of factors.
    """
    data=copy.deepcopy(objs)
    for news_item in data:
        for mention in news_item.entity_mentions:
            mention.identity='%s%s' % (prefix, mention.mention)
            if 'docid' in factors:
                mention.identity+=news_item.identifier.split('_')[-1]
            if 'type' in factors:
                mention.identity+=mention.the_type

    filename='bin/mention_%s_graph.pkl' % '_'.join(factors)

    with open(filename, 'wb') as w:
        pickle.dump(data, w)

    generate_graph(data, filename)

def generate_baseline_el_graphs(objs):
    """
    Generate baseline graphs.
    """
    generate_identity(objs, factors=[])
    generate_identity(objs, factors=['docid'])
    generate_identity(objs, factors=['type'])
    generate_identity(objs, factors=['docid', 'type'])
        

if __name__ == "__main__":

    input_dir='documents'
    pickle_file='bin/%s.pkl' % input_dir
    if os.path.isfile(pickle_file):
        print('pickle file exists. Loading it now...')
        with open(pickle_file, 'rb') as f:
            news_items_with_entities=pickle.load(f)
    else:
        print('Pickle file does not exist. Let us load the news items and run NER...')
        news_items=load_news_items(input_dir)
        news_items_with_entities=recognize_entities(news_items)
        with open('bin/%s.pkl' % input_dir, 'wb') as w:
            pickle.dump(news_items_with_entities, w)

    generate_baseline_el_graphs(news_items_with_entities)
    
