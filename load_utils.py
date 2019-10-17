import glob
import json
import pickle
import os.path

import algorithm_utils as algorithm
import config

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

def strip_identity(i):
    """
    Normalize identity to only contain the ID.
    """
    identity = i.replace('http://cltl.nl/entity#', '')
    return identity.replace(' ', '_')

# ------- Loading news items ----------------------

def map_offsets_to_tids(nlp, objects, ner_system):
    """Map entity mention offsets to token ids."""
    for news_item in objects:
        text = f"{news_item.title}\n{news_item.content}"
        text = text.strip()
        processed = nlp(text)
        if ner_system=='gold':
            entities=news_item.gold_entity_mentions
        else:
            entities=news_item.sys_entity_mentions
        for entity in entities:
            begin = int(entity.begin_offset)
            end = int(entity.end_offset)
            min_token_begin = None
            min_token_end = None
            min_begin = 999
            min_end = 999
            begin_sent = -1
            for sent_i, sent in enumerate(processed.sents):
                for token in sent:
                    begin_offset = token.idx
                    end_offset = token.idx + len(token.text)
                    if abs(begin - begin_offset) < min_begin:
                        min_begin = abs(begin - begin_offset)
                        min_token_begin = token.i
                        begin_sent = sent_index = str(sent_i + 1)
                    if abs(end - end_offset) < min_end:
                        min_end = abs(end - end_offset)
                        min_token_end = token.i
            entity.begin_index = min_token_begin
            entity.end_index = min_token_end
            tokens = list(range(min_token_begin, min_token_end + 1))
            entity.tokens = list(map(lambda x: f't{x}', tokens))
            entity.sentence = begin_sent
    return objects

def get_docs_with_entities(outdir, input_dir, nl_nlp, ner_system):
    """Obtain news items with recognized entities."""
    pkl_docs = '%s.pkl' % input_dir
    ent_addon = '_{}'.format(ner_system)
    pkl_docs_with_entities = '%s%s.pkl' % (input_dir, ent_addon)

    if os.path.isfile(pkl_docs_with_entities):
        print('pickle file with recognized entities exists. Loading it now...')
        news_items_with_entities = load_news_items(pkl_docs_with_entities)
    else:
        print('Pickle file does not exist. Let us load the news items and run NER...')
        news_items = load_news_items(pkl_docs)
        print('Loaded %d news items' % len(news_items))
        if ner_system == 'gold':
            news_items_with_entities = algorithm.recognize_entities_gold(news_items)
            news_items_with_entities = map_offsets_to_tids(nl_nlp, news_items_with_entities, ner_system)
        else:
            news_items_with_entities = algorithm.recognize_entities_spacy(nl_nlp, news_items)
        save_news_items(pkl_docs_with_entities,
                        news_items_with_entities)
    return news_items_with_entities
