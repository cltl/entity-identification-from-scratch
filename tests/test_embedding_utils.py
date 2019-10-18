from pytorch_pretrained_bert import BertModel, BertTokenizer

import embeddings_utils as emb_utils
import wip.embeddings as embu
from config import Config
import pickle_utils as pkl
import naf_utils as naf
from path import Path


def test_get_entity_and_sentence_embeddings():
    cfg = Config('cfg/test.yml')
    iteration = 1
    model = BertModel.from_pretrained(cfg.bert_model)
    tokenizer = BertTokenizer.from_pretrained(cfg.bert_model,
                                              do_lower_case=False)

    # TODO directly load news_items_with_entities from NAF
    news_items = pkl.load_news_items('%s.pkl' % cfg.input_dir)
    naf0 = Path('{}/0'.format(cfg.naf_dir))
    news_items_with_entities = naf.patch_classes_with_entities(news_items, naf0, cfg.naf_entity_layer)

    # TODO directly load news_items_with_entities from NAF
    entity_embeddings, sent_embeddings, all_emb = embu.get_entity_and_sentence_embeddings(cfg.naf_dir,
                                                                                               iteration,
                                                                                               model,
                                                                                               tokenizer,
                                                                                               news_items_with_entities,
                                                                                               cfg.naf_entity_layer,
                                                                                               modify_entities=cfg.modify_entities)
    doc1 = 'Brachymeria_eublemmae'
    doc2 = 'Tetrastichus_apanteles'
    assert doc2 not in entity_embeddings.keys()
    assert doc2 in sent_embeddings.keys()
    assert doc2 not in all_emb.keys()
    assert len(entity_embeddings[doc1].keys()) == 2
    assert len(entity_embeddings[doc1]['e1']) == 3072
    assert len(sent_embeddings[doc1].keys()) == 2
    assert sent_embeddings[doc1]['1'].shape[0] == 768
    assert len(all_emb[doc1]['e1']) == 3072 + 768
