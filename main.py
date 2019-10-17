import numpy as np
import copy
import nl_core_news_sm
import sys
from path import Path

from pytorch_pretrained_bert import BertTokenizer, BertModel
import Levenshtein as lev
from collections import defaultdict

import pickle_utils as pkl
import analysis_utils as analysis
import algorithm_utils as algorithm
import naf_utils as naf
import embeddings_utils as emb_utils
import config
import doc2vec


################## Run iteration 1 or 2 or more #########################

def similar(a, b, tau=0.8):
    """
    Check if two strings are similar enough.
    """
    return lev.ratio(a, b) >= tau


def is_abbrev(abbrev, text):
    abbrev = abbrev.lower()
    text = text.lower()
    words = text.split()
    if not abbrev:
        return True
    if abbrev and not text:
        return False
    if abbrev[0] != text[0]:
        return False
    elif words:
        return any(is_abbrev(abbrev[1:], text[i + 1:]) for i in range(len(words[0])))
    else:
        return False


def abbreviation(a, b):
    return is_abbrev(a, b) or is_abbrev(b, a)


def pregroup_clusters(news_items_with_entities):
    cands = []
    for item in news_items_with_entities:
        for e in item.sys_entity_mentions:
            mention = e.mention
            key = '%s#%s' % (item.identifier, e.eid)
            found = False
            for c in cands:
                if found:
                    break
                for other_mention, other_identifier in c:
                    if similar(other_mention, mention) or abbreviation(other_mention, mention):
                        c.append(tuple([mention, key]))
                        found = True
                        break

            if not found:
                new_c = tuple([mention, key])
                cands.append([new_c])
    return cands


def run_embeddings_system(data, embeddings, iteration, naf_folder, nl_nlp, graph_filename, sys_name):
    """
    Run the embeddings system.
    """
    refined_news_items = copy.deepcopy(data)
    print('Now pre-grouping clusters...')
    candidate_clusters = pregroup_clusters(refined_news_items)
    print('Clusters pre-grouped. We have %d groups.' % len(candidate_clusters))
    print(candidate_clusters)
    if sys_name == 'string_features':
        new_ids = {}
        for cid, members in enumerate(candidate_clusters):
            for mention, eid in members:
                new_ids[eid] = str(cid)
    else:  # sys_name=='embeddings'
        # m2id=algorithm.construct_m2id(refined_news_items)
        # print('M2ID', len(m2id.keys()))
        new_ids = algorithm.cluster_identities(candidate_clusters,
                                               embeddings,
                                               max_d=58)
    # assert len(new_ids)==len(ids), 'Mismatch between old and new ids. Old=%d; new=%d' % (len(ids),
    #                                                                                    len(new_ids))
    refined_news_items = algorithm.replace_identities(refined_news_items,
                                                      new_ids)

    # ANALYZE
    #    inspect_data(refined_news_items)

    naf_iter = naf_folder / str(iteration)
    naf.create_nafs(naf_iter,
                    refined_news_items,
                    nl_nlp,
                    cfg.corpus_uri)

    naf.add_ext_references_to_naf(refined_news_items,
                                  f'iteration{iteration}',
                                  cfg.naf_entity_layer,
                                  naf_folder / str(iteration - 1),
                                  naf_iter)

    algorithm.generate_graph(refined_news_items, graph_filename)

    ids = analysis.inspect_data(refined_news_items, graph_filename)

    return refined_news_items, new_ids


if __name__ == "__main__":

    cfg = config.create('cfg/abstracts_nif35.yml')

    print('Directories refreshed.')

    # LOAD MODELS

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained(cfg.bert_model,
                                              do_lower_case=False)
    print('BERT tokenizer loaded')

    # Load pre-trained model (weights)
    model = BertModel.from_pretrained(cfg.bert_model)
    print('BERT model loaded')

    nl_nlp = nl_core_news_sm.load()

    doc2vec_model = doc2vec.get_doc2vec_model(cfg.emb_dir, cfg.input_dir, force=False)
    print(doc2vec_model.docvecs[0].shape)
    print('Doc2Vec model loaded')

    # ------ Generate NAFs and fill classes with entity mentions (Steps 1 and 2) --------------------

    # TODO: COMBINE NAFs with classes processing to run spacy only once!
    news_items = pkl.load_news_items('%s.pkl' % cfg.input_dir)

    #naf_empty = Path('{}/empty'.format(cfg.naf_dir))
    #naf.create_nafs(naf_empty, news_items, nl_nlp, cfg.corpus_uri, cfg.ner)

    naf0 = Path('{}/0'.format(cfg.naf_dir))  # NAF folder before iteration 1
    naf.create_nafs(naf0, news_items, nl_nlp, cfg.corpus_uri, cfg.ner)
    #if cfg.ner == 'gold':
    #naf.add_mentions_to_naf(news_items,
    #                              cfg.ner,
    #                              cfg.naf_entity_layer,
    #                              naf_empty,
    #                              naf0)
    news_items_with_entities=naf.patch_classes_with_entities(news_items, naf0, cfg.naf_entity_layer)
    # ------- Run embeddings system -----------------

    iteration = 1
    # BERT embeddings
    entity_embeddings, sent_embeddings, all_emb = emb_utils.get_entity_and_sentence_embeddings(cfg.naf_dir,
                                                                                               iteration,
                                                                                               model,
                                                                                               tokenizer,
                                                                                               news_items_with_entities,
                                                                                               cfg.naf_entity_layer,
                                                                                               modify_entities=cfg.modify_entities)
    print('Entity and sentence embeddings created')

    # TODO define method to check created embeddings
    # print(sent_embeddings[test_key].keys())
    # print(entity_embeddings[test_key].keys())
    # print(all_emb[test_key].keys())
    # print(sent_embeddings[test_key]['1'].shape)
    # print(entity_embeddings[test_key]['e1'].shape)
    # print(all_emb[test_key]['e1'].shape)
    # id_embeddings=emb_utils.sent_to_id_embeddings(sent_embeddings,
    #                                              data)

    # integrate doc2vec into embeddings
    full_embeddings = defaultdict(dict)
    for i, item in enumerate(news_items_with_entities):
        doc_vector = doc2vec_model.docvecs[i]
        current_emb = all_emb[item.identifier]
        for eid, embs in current_emb.items():
            full_embeddings[item.identifier][eid] = np.concatenate((embs, doc_vector), axis=0)

    #print('Full embedding shape', full_embeddings[test_key]['e2'].shape)
    if not full_embeddings:
        raise ValueError('Full embeddings are empty. Refusing to continue')

    data, ids = run_embeddings_system(news_items_with_entities,
                                      full_embeddings,
                                      iteration,
                                      Path(cfg.naf_dir),
                                      nl_nlp,
                                      cfg.graphs_file,
                                      cfg.sys_name)
