import pathlib
import glob
from KafNafParserPy import KafNafParser
import re
from KafNafParserPy.header_data import CHeader, CfileDesc, Clp
from KafNafParserPy.external_references_data import CexternalReference
import time
import nl_core_news_sm
from classes import NewsItem, EntityMention


processor_name = "Entity detection for historical Dutch"


def load_news_items_with_entities(naf_dir):
    """Load news items from NAF."""
    news_items = []
    # load NAF files from the previous iteration
    file_names = glob.glob("{}/*.naf".format(naf_dir))
    for doc_index, f in enumerate(file_names):
        news_item = create_news_item_naf(f)
        news_items.append(news_item)
    return news_items 


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
    entities = []
    if naf.entity_layer is not None:
        entities = [create_entity_mention(e, naf) for e in naf.entity_layer]
    return NewsItem(
        dct=dct,
        title=title,
        collection=collection,
        identifier=identifier,
        content=naf.raw,
        sys_entity_mentions=entities,
        gold_entity_mentions=[]
    )


def get_sentences(naf_file):
    naf = KafNafParser(naf_file)
    i = 1
    sentences = []
    s = []
    for wf in naf.text_layer:
        if int(wf.get_sent()) == i:
            s.append(wf.get_text())
        else:
            sentences.append(" ".join(s))
            i += 1
            s = [wf.get_text()]
    sentences.append(" ".join(s))
    return sentences


def create_header(news_item):
    header = CHeader(type="NAF")
    file_desc = CfileDesc()
    if news_item.dct is None:
        news_item.dct = time.strftime('%Y-%m-%dT%H:%M:%S%Z')
    file_desc.set_creationtime(news_item.dct)
    if news_item.title:
        file_desc.set_title(news_item.title)
    header.set_fileDesc(file_desc)
    header.set_uri("http://www.{}.nl/{}".format(news_item.collection, news_item.identifier))
    return header


def create_naf_from_item(news_item):
    naf = KafNafParser(type="NAF")
    naf.set_version("3.0")
    naf.set_language("nl")
    naf.set_header(create_header(news_item))
    naf.add_linguistic_processor('raw', create_linguistic_processor())
    naf.set_raw(news_item.content)
    return naf


def write_naf(naf, file_out):
    naf.dump(file_out)


def inject_spacy(naf, doc):
    naf.add_linguistic_processor('text', create_linguistic_processor())
    naf.add_linguistic_processor('terms', create_linguistic_processor())
    naf.add_linguistic_processor('entities', create_linguistic_processor())
    # word forms
    term_at = {}
    ending_at = {}
    for s_i, sentence in enumerate(doc.sents, 1):
        for wf in sentence:
            token = naf.create_wf(wf.text, str(s_i), wf.idx)
            # terms
            term = naf.create_term(wf.lemma_, wf.pos_, wf.tag_, [token])
            term_at[wf.idx] = term.get_id()
            ending_at[wf.idx + len(wf.text)] = term.get_id()

    # entities; mapping to terms
    term_pfx = re.search(r"\D+", list(term_at.values())[0]).group()
    for ent in doc.ents:
        term_ids = get_term_ids(term_at[ent.start_char], ending_at[ent.end_char], term_pfx)
        naf.create_entity(ent.label_, term_ids)
    return naf


def get_term_ids(start_term, end_term, pfx):
    start_id = get_index(start_term)
    end_id = get_index(end_term) + 1
    return ["{}{}".format(pfx, i) for i in range(start_id, end_id)]


def get_index(term_id):
    return int(re.search(r"\d+", term_id).group())


def create_linguistic_processor():
    lp = Clp()
    lp.set_name(processor_name)
    lp.set_version("1.0")
    lp.set_timestamp()
    return lp


def run_spacy_and_write_to_naf(news_items, naf_dir):
    spacy_nl = nl_core_news_sm.load()

    for item in news_items:
        naf = create_naf_from_item(item)
        # item.content may be Literal (from nif corpus creation)
        naf = inject_spacy(naf, spacy_nl(str(item.content)))
        write_naf(naf, f_name(naf_dir, item))


def load_naf(naf_dir, item):
    filename = f_name(naf_dir, item)
    # print(filename)
    return KafNafParser(filename)


def f_name(naf_dir, item):
    return "{}/{}.naf".format(naf_dir, item.identifier)


def add_ext_references(refined_news_items, naf_dir0, naf_dir1):
    for item in refined_news_items:
        naf = load_naf(naf_dir0, item)
        if item.sys_entity_mentions and naf.entity_layer is not None:
            for e_naf, e_ref in zip(naf.entity_layer, item.sys_entity_mentions):
                external_ref = CexternalReference()
                external_ref.set_reference(e_ref.identity)
                external_ref.set_source('iteration1')
                e_naf.add_external_reference(external_ref)
            write_naf(naf, f_name(naf_dir1, item))


def add_ext_references_gold(refined_news_items, naf_dir0, naf_dir1):
    for item in refined_news_items:
        naf = load_naf(naf_dir0, item)
        if item.gold_entity_mentions and naf.entity_layer is not None:
            for e_naf, e_ref in zip(naf.entity_layer, item.gold_entity_mentions):
                external_ref = CexternalReference()
                external_ref.set_reference(e_ref.identity)
                external_ref.set_source('gold')
                e_naf.add_external_reference(external_ref)
            write_naf(naf, f_name(naf_dir1, item))
