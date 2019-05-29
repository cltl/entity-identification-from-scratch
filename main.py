import classes

import networkx as nx
import copy
import os.path
from pprint import pprint
import glob
import json
import pickle
import spacy
import sys
import wikitextparser as wtp

def shift_all(links_json, x):
    new_json={}
    for start, end in links_json.keys():
        new_start=start-x
        new_end=end-x
        new_json[tuple([new_start, new_end])]=links_json[(start, end)]
    return new_json

def get_text_and_links(wikitext):
    parsed = wtp.parse(wikitext)
    basic_info=parsed.sections[0]
    saved_links={}

    num_links=len(basic_info.wikilinks)
    for i in range(num_links):
        index=num_links-i-1
        link=basic_info.wikilinks[index]
        original_span=link.span
        start=original_span[0]
        end=original_span[1]
        target=link.target
        text=link.text
        if not target.startswith('w:'):
            basic_info[start:end]=""
            move_to_left=end-start
        else:
            basic_info[original_span[0]:original_span[1]]=text
            move_to_left=end-start-len(text)
        saved_links=shift_all(saved_links, move_to_left)
        if target.startswith('w:'):
            new_end=end-move_to_left
            saved_links[tuple([start, new_end])]=target

    return basic_info, saved_links

def create_gold_mentions(links, text):
    mentions=[]
    for offset, meaning in links.items():
        start, end=offset
        mention=text[start:end]
        obj=classes.EntityMention(
            mention=mention,
            begin_index=start,
            end_index=end,
            identity=meaning
        )
        mentions.append(obj)
    return mentions

def load_news_items(loc):
    """
    Load news items into objects defined in the classes file.
    """
    
    news_items=set()
    for file in glob.glob('%s/*.json' % loc):
        with open(file, 'r') as f:
            data=json.load(f)
        text, links=get_text_and_links(data['body'])
        news_item_obj=classes.NewsItem(
            content=text,
            title=data['title'],
            identifier=file.split('.')[0],
            collection=loc,
            gold_entity_mentions=create_gold_mentions(links, text)
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
            news_item.sys_entity_mentions.append(ent_mention_obj)
        print(i)
    return news_items

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

def generate_identity(objs,prefix='http://cltl.nl/entity#', factors=[]):
    """
    Decide which entities are identical, based on a set of recognized entity mentions and flexible set of factors.
    """
    data=copy.deepcopy(objs)
    for news_item in data:
        for mention in news_item.sys_entity_mentions:
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
    
