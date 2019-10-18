from collections import defaultdict
import glob
import embeddings_utils as embu
import numpy as np
from corpus_handler import naf_handler as naf


def get_entity_and_sentence_embeddings(naf_dir, iteration, model, tokenizer, news_items, naf_entity_layer, modify_entities=False):
    """
    Obtain entity and sentence embeddings using BERT for an entire NAF collection.
    """
    sent_emb = defaultdict(dict)
    ent_emb = defaultdict(dict)
    concat_emb = defaultdict(dict)

    # load NAF files from the previous iteration
    file_names = glob.glob('%s/%d/*.naf' % (naf_dir, iteration - 1))
    nb_files = len(file_names)
    mark = 1
    while mark < nb_files / 10:
        mark *= 10
    doc_index = 1
    for f in file_names:
        if doc_index % mark == 0:
            print("getting embeddings: {}/{}".format(doc_index, nb_files))

        s = naf.get_sentences(f)
        news_item = naf.create_news_item_naf(f)
        doc_id = news_item.identifier
        entities = news_item.sys_entity_mentions

        offset = 0
        # Sentence per sentence: run BERT, extract sentence embeddings, extract entity embeddings, and concatenate
        for index, sentence in enumerate(s):
            sent_index = str(index + 1)
            verbose = doc_index % mark == 0 and index == 0

            tokenized_text, encoded_layers = embu.get_bert_embeddings(sentence, model, tokenizer)

            # Get sentence embeddings
            sentence_embeddings = embu.get_bert_sentence_embeddings(encoded_layers)
            if verbose:
                print('Sentence embedding shape', sentence_embeddings.shape[0])
            sent_emb[doc_id][sent_index] = sentence_embeddings

            # Concatenated word embeddings
            word_embeddings = embu.get_bert_word_embeddings(tokenized_text, encoded_layers)
            if verbose:
                print('Word embeddings shape', len(word_embeddings), len(word_embeddings[0]))

            verbose = False
            entity_embeddings, new_offset = embu.map_bert_embeddings_to_tokens(tokenized_text,
                                                                               entities,
                                                                               word_embeddings,
                                                                               index + 1,
                                                                               offset,
                                                                               verbose)
            # concat_emb maps documents to entities and
            # associates each entity with its embedding and the embedding of the sentence
            for eids, embs in entity_embeddings.items():
                ent_emb[doc_id][eids] = embs
                concat_emb[doc_id][eids] = np.concatenate((embs, sentence_embeddings), axis=0)
            offset += new_offset

        doc_index += 1
    print('finished extracting embeddings.')
    entity_count = 0
    for v in concat_emb.values():
        entity_count += len(v)
    print('Collected {} entity embeddings over {} documents'.format(entity_count, doc_index - 1))

    return ent_emb, sent_emb, concat_emb
