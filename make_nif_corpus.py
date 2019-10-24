from rdflib import Graph, URIRef

import pickle_utils as pkl
from config import Config
import classes

# ------ NIF datasets loader ---------------------

def load_article_from_nif_file(nif_file, corpus_name, limit=1000000):
    """
    Load a dataset in NIF format.
    """
    print(f'NOW LOADING THE NIF FILE {nif_file}')
    g = Graph()
    #for nif_file in glob.glob('%s/*.ttl' % nif_dir):
    g.parse(nif_file, format="n3")

    print(f'THE FILE {nif_file} IS LOADED. NOW QUERYING')

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
        query = """ SELECT ?id ?mention ?start ?end ?gold
        WHERE {
                ?id nif:anchorOf ?mention ;
                nif:beginIndex ?start ;
                nif:endIndex ?end ;
                nif:referenceContext <%s> .
                OPTIONAL { ?id itsrdf:taIdentRef ?gold . }
        } ORDER BY ?start""" % str(article['articleid'])
        qres_entities = g.query(query)
        all_entities=[]
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
            all_entities.append(entity_obj)
        news_item_obj = classes.NewsItem(
            content=article['string'],
            identifier=doc_id,  # "http://yovisto.com/resource/dataset/iswc2015/doc/281#char=0,4239",
            dct=article['date'],
            collection=corpus_name,
            title='',
            gold_entity_mentions=all_entities
        )
        print(article['articleid'], len(news_item_obj.gold_entity_mentions))
        news_items.add(news_item_obj)
    return news_items

# Specify your config file here:
config_file='cfg/abstracts50.yml'

cfg = Config(config_file)
cfg.setup_input()

# Load configuration variables
# min_length = cfg.min_text_length

# Load a number of news items from a NIF file
news_items = load_article_from_nif_file(cfg.raw_input,
                                        cfg.corpus_name,
                                        limit=cfg.max_documents or 1000000)

# Save the news articles to pickle
print(cfg.news_items_file())
pkl.save_news_items(cfg.news_items_file(), news_items)
