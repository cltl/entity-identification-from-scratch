from pytorch_pretrained_bert import BertModel, BertTokenizer

import doc2vec
import main
import wip.embeddings as embu
from config import Config
import pickle_utils as pkl
import naf_utils as naf
from path import Path


def test_get_entity_and_sentence_embeddings():
    cfg = Config('cfg/test.yml')
    model = BertModel.from_pretrained(cfg.bert_model)
    tokenizer = BertTokenizer.from_pretrained(cfg.bert_model,
                                              do_lower_case=False)
    doc2vec_model = doc2vec.get_doc2vec_model(cfg.emb_dir, cfg.input_dir, force=False)

    full_embeddings, news_items = embu.get_entity_and_sentence_embeddings(
        "{}/0".format(cfg.naf_dir),
        model,
        tokenizer,
        doc2vec_model
        )
    doc1 = 'Brachymeria_eublemmae'
    doc2 = 'Tetrastichus_apanteles'
    enty_embs = 3072
    sent_embs = 768
    doc_embs = 1000
    assert doc2 not in full_embeddings.keys()

    assert len(full_embeddings[doc1].keys()) == 2
    assert full_embeddings[doc1]['e1'].shape[0] == enty_embs + sent_embs + doc_embs
