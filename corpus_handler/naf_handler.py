from KafNafParserPy import KafNafParser
from classes import NewsItem, EntityMention
import re


def get_header_attributes_naf(naf):
    """returns: doc creation time, title, collection, doc id"""
    header = naf.header
    title = header.get_fileDesc().get_title()
    uri = header.get_uri()
    m = re.match(r"http://(.*)\.[a-z][a-z]/(.*)\/*", uri)
    return header.get_dct(), title, m.group(1).replace('www.', ''), m.group(2)


def get_terms(e, terms_layer):
    terms = []
    for r in e.get_references():
        for t in r.get_span():
            terms.append(terms_layer.get_term(t.get_id()))
    return terms


def to_word_forms(terms, text_layer):
    return [text_layer.get_wf(t.get_span_ids()[0]) for t in terms]


def create_entity_mention(e, naf):
    identity = None
    for r in e.get_external_references():
        identity = r.get_reference()

    terms = get_terms(e, naf.term_layer)
    term_ids = [t.get_id() for t in terms]
    wfs = to_word_forms(terms, naf.text_layer)

    return EntityMention(
        eid=e.get_id(),
        sentence=int(wfs[0].get_sent()),  # e.g. 4 -> which sentence is the entity mentioned in
        mention=" ".join([wf.get_text() for wf in wfs]),  # e.g. "John Smith" -> the mention of an entity as found in text
        the_type=e.get_type(),
        begin_index=int(re.search(r"\d+", term_ids[0]).group(0)),  # e.g. 3 -> begin token id
        end_index=int(re.search(r"\d+", term_ids[-1]).group(0)),  # e.g. 5 -> end token id
        begin_offset=int(wfs[0].get_offset()),  # e.g. 15 -> begin offset
        end_offset=int(wfs[-1].get_offset()) + int(wfs[-1].get_length()),  # e.g. 25 -> end offset
        identity=identity,  # gold link if existing
        tokens=term_ids
        )


def create_news_item_naf(naf_file):
    naf = KafNafParser(naf_file)
    dct, title, collection, identifier = get_header_attributes_naf(naf)
    return NewsItem(
        dct=dct,
        title=title,
        collection=collection,
        identifier=identifier,
        content=naf.raw,
        sys_entity_mentions=[create_entity_mention(e, naf) for e in naf.entity_layer],
        gold_entity_mentions=[]
    )
