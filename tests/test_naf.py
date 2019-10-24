from wip.naf_handler import inject_spacy, create_news_item_naf, write_naf, create_naf_from_item
import nl_core_news_sm


def test_load_naf_no_ext_refs():
    """reads a naf file and creates a news item"""
    naf_file = 'tests/data/test_no_ext_refs.naf'

    news_item = create_news_item_naf(naf_file)
    assert news_item.identifier == 'example-news.html'
    assert news_item.content == "onderkoopman Willem Adriaan Palm is in afwachting van werk naar Semarang gestuurd."
    assert news_item.title is None
    assert news_item.collection == 'newsreader-project'
    assert news_item.dct == '2019-09-13T11:32:38+0200'
    assert not news_item.gold_entity_mentions
    assert len(news_item.sys_entity_mentions) == 2

    entity = news_item.sys_entity_mentions[0]
    assert entity.eid == 'e1'
    assert entity.sentence == 1
    assert entity.mention == "Willem Adriaan Palm"
    assert entity.the_type == "PER"
    assert entity.begin_index == 1
    assert entity.end_index == 3
    assert entity.begin_offset == 13
    assert entity.end_offset == 32  # begin offset + length of last index (TODO check)
    assert entity.identity is None
    assert entity.tokens == ['t_1', 't_2', 't_3']


def test_load_naf_ext_refs():
    """reads a naf file and creates a news item"""
    naf_file = 'tests/data/test_ext_refs.naf'

    news_item = create_news_item_naf(naf_file)
    assert news_item.identifier == 'wiki_1'
    assert news_item.title == '"Crocodile Hunter" Steve Irwin omgekomen in Australië'
    assert news_item.collection == 'wikinews2'
    assert not news_item.gold_entity_mentions
    assert len(news_item.sys_entity_mentions) == 25

    entity = news_item.sys_entity_mentions[0]
    assert entity.eid == 'e1'
    assert entity.sentence == 1
    assert entity.mention == '" Crocodile'
    assert entity.the_type == "PER"
    assert entity.begin_index == 1
    assert entity.end_index == 2
    assert entity.begin_offset == 0
    assert entity.end_offset == 10
    assert entity.identity == '0_3'
    assert entity.tokens == ['t1', 't2']


def test_load_naf_empty_ext_refs():
    naf_file = 'tests/data/test_empty_ext_ref.naf'

    news_item = create_news_item_naf(naf_file)
    entity = news_item.sys_entity_mentions[0]
    assert entity.identity is None


def is_same_entity(e_spacy, e_mention):
    return e_spacy.text == e_mention.mention \
           and e_spacy.start_char == e_mention.begin_offset \
           and e_spacy.end_char == e_mention.end_offset \
           and e_spacy.label_ == e_mention.the_type


def test_spacy_entities():
    naf_file = 'tests/data/test_ext_refs.naf'
    news_item = create_news_item_naf(naf_file)
    nl_nlp = nl_core_news_sm.load()
    spacy_doc = nl_nlp(news_item.content)
    assert len(spacy_doc.ents) == len(news_item.sys_entity_mentions)
    assert spacy_doc.ents[0].text == "Crocodile Hunter"

    naf = create_naf_from_item(news_item)
    naf = inject_spacy(naf, spacy_doc)
    naf_out = 'tests/data/test_ext_refs3.naf'
    write_naf(naf, naf_out)
    news_item2 = create_news_item_naf(naf_out)

    assert news_item2.sys_entity_mentions[0].mention == "Crocodile Hunter"
    assert is_same_entity(spacy_doc.ents[0], news_item2.sys_entity_mentions[0])


def test_naf2items2naf():
    naf_file = 'tests/data/test_ext_refs.naf'
    news_item = create_news_item_naf(naf_file)
    out_naf = 'tests/data/test_ext_refs2.naf'
    write_naf(create_naf_from_item(news_item), out_naf)
    news_item2 = create_news_item_naf(out_naf)
    assert news_item.has_same_header_and_content(news_item2)
