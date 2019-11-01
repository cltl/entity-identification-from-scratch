from collections import defaultdict
import glob
import embeddings_utils as embu
import wip.align_tokens as align
import numpy as np
from wip import naf_handler as naf
from pytorch_pretrained_bert import BertTokenizer, BertModel
import doc2vec


def load_models(cfg):
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained(cfg.bert_model, do_lower_case=False)
    print('BERT tokenizer loaded')

    # Load pre-trained model (weights)
    model = BertModel.from_pretrained(cfg.bert_model)
    print('BERT model loaded')

    d2v = doc2vec.get_doc2vec_model(cfg, force=False)
    print(d2v.docvecs[0].shape)
    print('Doc2Vec model loaded')
    return tokenizer, model, d2v


def get_entity_and_sentence_embeddings(naf_dir, model, tokenizer, doc2vec_model):
    """
    Obtain entity and sentence embeddings using BERT for an entire NAF collection.
    """
    concat_emb = defaultdict(dict)
    full_embeddings = defaultdict(dict)
    news_items = []
    # load NAF files from the previous iteration
    file_names = glob.glob("{}/*.naf".format(naf_dir))
    nb_files = len(file_names)
    mark = 1
    while mark < nb_files / 10:
        mark *= 10
    for doc_index, f in enumerate(file_names):
        if doc_index % mark == 0:
            print("getting embeddings: {}/{}".format(doc_index, nb_files))

        sentences, sentence_tokens = naf.get_sentences(f)
        news_item = naf.create_news_item_naf(f)
        news_items.append(news_item)
        doc_id = news_item.identifier
        entities = news_item.sys_entity_mentions


        offset=1 # offset for sentences after the first one. Starts at 1 because the token IDs start at 1 and not 0.

        # Sentence per sentence: run BERT, extract sentence embeddings, extract entity embeddings, and concatenate
        for index, sentence in enumerate(sentences, start=1):
            verbose = doc_index % mark == 0 and index == 1

            word_embeddings, sentence_embeddings, bert_tokens = get_bert_word_and_sentence_embeddings(model, sentence, tokenizer, verbose)

            our_tokens=sentence_tokens[index-1]

            entity_embeddings = align.map_bert_embeddings_to_tokens(bert_tokens,
                                                                    our_tokens,
                                                                    entities,
                                                                    word_embeddings,
                                                                    index,
                                                                    offset,
                                                                    verbose)
            offset+=len(our_tokens)
            # concat_emb maps documents to entities and
            # associates each entity with its embedding and the embedding of the sentence
            for e_id, emb in entity_embeddings.items():
                concat_emb[doc_id][e_id] = np.concatenate((emb, sentence_embeddings), axis=0)

        if doc_id in concat_emb.keys():
            for e_id, emb in concat_emb[doc_id].items():
                full_embeddings[doc_id][e_id] = np.concatenate((emb, doc2vec_model.docvecs[doc_index]), axis=0)

    print('finished extracting embeddings.')
    entity_count = 0
    for v in concat_emb.values():
        entity_count += len(v)
    print('Collected {} entity embeddings over {} documents'.format(entity_count, doc_index))
    return full_embeddings, news_items


def get_bert_word_and_sentence_embeddings(model, sentence, tokenizer, verbose=False):
    """Obtain word and sentence embeddings from BERT."""
    tokenized_text, encoded_layers = embu.get_bert_embeddings(sentence, model, tokenizer)
    print(tokenized_text)
    # Get sentence embeddings
    sentence_embeddings = embu.get_bert_sentence_embeddings(encoded_layers)
    if verbose:
        print('Sentence embedding shape', sentence_embeddings.shape[0])
    # Concatenated word embeddings
    word_embeddings = embu.get_bert_word_embeddings(tokenized_text, encoded_layers)
    if verbose:
        print('Word embeddings shape', len(word_embeddings), len(word_embeddings[0]))
    return word_embeddings, sentence_embeddings, tokenized_text
