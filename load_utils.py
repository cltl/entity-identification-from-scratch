import wikitextparser as wtp
import glob
import json
import re
import pickle
import os.path

import algorithm_utils as algorithm
import classes

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

def clean_wiki(wikitext):
    """Removes wiki flags"""
    text = str(wikitext)
    # date tags {{Datum|foo}}
    text = re.sub(r'\{\{Datum\|(.*)\}\}', r'\1.', text)
    # wiki entities {{W|foo}}
    text = re.sub(r'\{\{W\|([\|]*)\}\}', r'\1', text)
    # wiki entities {{w|id|foo}}
    text = re.sub(r'\{\{w\|[^\|]*\|([^\|]*)\}\}', r'\1', text)
    # wiki non Dutch entities {{w|id|foo|lang}}
    text = re.sub(r'\{\{w\|[^\|]*\|([^\|]*)\|[^\|]*\}\}', r'\1', text)
    # base markup {{foo}}
    text = re.sub(r'\{\{([^\|]*)\}\}', r'\1', text)
    # anything else {{bla}} is deleted
    text = re.sub(r'\{\{([^\}]*)\}\}', '', text)
    #text = re.split('\s+', text)
    return text


def strip_identity(i):
    identity=i.replace('http://cltl.nl/entity#', '')
    return identity.replace(' ', '_')

# ------ Processing news items -------------------


def load_news_items(a_file):
    """Loads news items with entities."""
    with open(a_file, 'rb') as f:
        news_items_with_entities = pickle.load(f)
    return news_items_with_entities

def save_news_items(a_file, data):
    """Save news items with entities."""
    with open(a_file, 'wb') as w:
        pickle.dump(data, w)
        
# ------- Loading news items ----------------------

def get_docs_with_entities(outdir, input_dir, nl_nlp):
    """Obtain news items processed with NER."""
    pkl_docs='%s.pkl' % input_dir
    ent_addon='_with_ent'
    pkl_docs_with_entities='%s%s.pkl' % (input_dir, ent_addon)
    
    if os.path.isfile(pkl_docs_with_entities):
        print('pickle file with recognized entities exists. Loading it now...')
        news_items_with_entities=load_news_items(pkl_docs_with_entities)
        
    else:
        print('Pickle file does not exist. Let us load the news items and run NER...')
        news_items=load_news_items(pkl_docs)
        print('Loaded %d news items' % len(news_items))
        news_items_with_entities=algorithm.recognize_entities(nl_nlp, news_items)
        save_news_items(pkl_docs_with_entities, 
                        news_items_with_entities)
    return news_items_with_entities
