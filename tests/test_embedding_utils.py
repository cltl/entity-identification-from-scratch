import wip.embeddings as embu
from config import Config
import pickle_utils as pkl
import wip.naf_handler as nafh


def test_get_entity_and_sentence_embeddings():
    cfg = Config('tests/cfg/test.yml')
    cfg.create_sysdirs()

    news_items = pkl.load_news_items(cfg.news_items_file())
    assert len(news_items) == 2

    assert cfg.create_input_nafs
    if cfg.create_input_nafs:
        nafh.run_spacy_and_write_to_naf(news_items, cfg.this_naf_indir())

    tokenizer, model, d2v = embu.load_models(cfg)
    full_embeddings, news_items_with_entities = embu.get_entity_and_sentence_embeddings(
        cfg.this_naf_indir(),
        model,
        tokenizer,
        d2v)

    doc1 = 'Brachymeria_eublemmae'
    doc2 = 'Tetrastichus_apanteles'
    enty_embs = 3072
    sent_embs = 768
    doc_embs = 1000
    assert doc2 in full_embeddings.keys()

    assert len(full_embeddings[doc1].keys()) == 2
    assert full_embeddings[doc1]['e1'].shape[0] == enty_embs + sent_embs + doc_embs

#def test_map_bert_embeddings_to_tokens():
    # TODO implement this, after we have the new mapping function
#    cfg = Config('')

#def test_new_mapping(sentence, verbose=True):
#    model = BertModel.from_pretrained(cfg.bert_model)
#    tokenizer = BertTokenizer.from_pretrained(cfg.bert_model,
#                                              do_lower_case=False)
#    word_embeddings, sentence_embeddings, bert_tokens = get_bert_word_and_sentence_embeddings(model, sentence, tokenizer, verbose)
#    console.log(word_embeddings.shape)


#sentence="After 5 months and 48 games , the match was abandoned in controversial circumstances with Karpov leading five wins to three ( with 40 draws ) , and replayed in the World Chess Championship 1985 ."
#test_new_mapping(sentence)
