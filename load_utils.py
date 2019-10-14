import wikitextparser as wtp
import glob
import json
import re
import pickle
import os.path
from rdflib import Graph, URIRef
from deprecated import deprecated

import algorithm_utils as algorithm
import classes
import config


# ------ Media Wiki processing utils. ----- #

@deprecated(reason="This kind of Wiki processing is not sufficiently reliable.")
def shift_all(links_json, x):
    """
    Shift the full text to account for the link markers.
    """
    new_json = {}
    for start, end in links_json.keys():
        new_start = start - x
        new_end = end - x
        new_json[tuple([new_start, new_end])] = links_json[(start, end)]
    return new_json


def get_text_and_links(wikitext):
    """
    Obtain text and links from a wikipedia text.
    """
    parsed = wtp.parse(wikitext)
    basic_info = parsed.sections[0]
    saved_links = {}

    num_links = len(basic_info.wikilinks)
    for i in range(num_links):
        index = num_links - i - 1
        link = basic_info.wikilinks[index]
        original_span = link.span
        start = original_span[0]
        end = original_span[1]
        target = link.target
        text = link.text
        if not target.startswith('w:'):
            basic_info[start:end] = ""
            move_to_left = end - start
        else:
            basic_info[original_span[0]:original_span[1]] = text
            move_to_left = end - start - len(text)
        saved_links = shift_all(saved_links, move_to_left)
        if target.startswith('w:'):
            new_end = end - move_to_left
            saved_links[tuple([start, new_end])] = target

    return basic_info, saved_links


@deprecated(reason="This kind of Wiki processing is not sufficiently reliable.")
def create_gold_mentions(links, text):
    """
    Create gold mentions from inline links in wikipedia.
    """
    mentions = []
    for offset, meaning in links.items():
        start, end = offset
        mention = text[start:end]
        obj = classes.EntityMention(
            mention=mention,
            begin_index=start,
            end_index=end,
            identity=meaning
        )
        mentions.append(obj)
    return mentions


def clean_wiki(wikitext):
    """Remove media wiki style and template markers."""
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
    # text = re.split('\s+', text)
    return text


def strip_identity(i):
    """
    Normalize identity to only contain the ID.
    """
    identity = i.replace('http://cltl.nl/entity#', '')
    return identity.replace(' ', '_')


# ------ NIF datasets loader ---------------------

def load_article_from_nif_files(nif_dir, limit=1000000, collection='wes2015'):
    """
    Load a dataset in NIF format.
    """
    print('NOW LOADING THE NIF FILES')
    g = Graph()
    for nif_file in glob.glob('%s/*.ttl' % nif_dir):
        g.parse(nif_file, format="n3")

    print('ALL FILES LOADED. NOW QUERYING')

    news_items = set()

    articles = g.query(
        """ SELECT ?articleid ?date ?string
    WHERE {
            ?articleid nif:isString ?string .
            OPTIONAL { ?articleid <http://purl.org/dc/elements/1.1/date> ?date . }
    }
    LIMIT %d""" % limit)
    for article in articles:
        doc_id = article['articleid'].replace('http://nl.dbpedia.org/resource/', '').split('/')[0]
        news_item_obj = classes.NewsItem(
            content=article['string'],
            identifier=doc_id,  # "http://yovisto.com/resource/dataset/iswc2015/doc/281#char=0,4239",
            dct=article['date'],
            collection=config.corpus_name,
            title=''
        )
        query = """ SELECT ?id ?mention ?start ?end ?gold
        WHERE {
                ?id nif:anchorOf ?mention ;
                nif:beginIndex ?start ;
                nif:endIndex ?end ;
                nif:referenceContext <%s> .
                OPTIONAL { ?id itsrdf:taIdentRef ?gold . }
        } ORDER BY ?start""" % str(article['articleid'])
        qres_entities = g.query(query)
        for eid, entity in enumerate(qres_entities):
            gold_link = str(entity['gold'])  # utils.getLinkRedirect(utils.normalizeURL(str(entity['gold'])))
            if gold_link.startswith('http://aksw.org/notInWiki'):
                gold_link = '--NME--'
            entity_obj = classes.EntityMention(
                begin_offset=int(entity['start']),
                end_offset=int(entity['end']),
                mention=str(entity['mention']),
                identity=gold_link,
                eid=f'e{eid}'
            )
            news_item_obj.gold_entity_mentions.append(entity_obj)
        news_items.add(news_item_obj)
    return news_items


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
