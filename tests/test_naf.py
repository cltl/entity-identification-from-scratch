from corpus_handler.naf_handler import create_news_item_naf


def test_load_naf_no_ext_refs():
    """reads a naf file and creates a news item"""
    naf_file = 'data/test_no_ext_refs.naf'

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
    naf_file = 'data/test_ext_refs.naf'

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
    naf_file = 'data/test_empty_ext_ref.naf'

    news_item = create_news_item_naf(naf_file)
    entity = news_item.sys_entity_mentions[0]
    assert entity.identity is None
