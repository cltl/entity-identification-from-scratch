import numpy as np
import shutil
import copy
import nl_core_news_sm
import sys
from path import Path
import os.path
from pytorch_pretrained_bert import BertTokenizer, BertModel
import Levenshtein as lev
from collections import defaultdict

import load_utils
import analysis_utils as analysis
import algorithm_utils as algorithm
import naf_utils as naf
import embeddings_utils as emb_utils
import config
import doc2vec

################## Run iteration 1 or 2 or more #########################

def similar(a,b, tau=0.8):
    """
    Check if two strings are similar enough.
    """
    return lev.ratio(a,b)>=tau

def is_abbrev(abbrev, text):
    abbrev=abbrev.lower()
    text=text.lower()
    words=text.split()
    if not abbrev:
        return True
    if abbrev and not text:
        return False
    if abbrev[0]!=text[0]:
        return False
    else:
        return (is_abbrev(abbrev[1:],' '.join(words[1:])) or
                any(is_abbrev(abbrev[1:],text[i+1:])
                    for i in range(len(words[0]))))

def abbreviation(a,b):
    return is_abbrev(a,b) or is_abbrev(b,a)


def pregroup_clusters(news_items_with_entities):
    cands=[]
    for item in news_items_with_entities:
        for e in item.sys_entity_mentions:
            mention=e.mention
            key='%s#%s' % (item.identifier, e.eid)
            found=False
            for c in cands:
                if found: break
                for other_mention, other_identifier in c:
                    if similar(other_mention, mention) or abbreviation(other_mention, mention):
                        c.append(tuple([mention, key]))
                        found=True
                        break
            if not found:
                new_c=tuple([mention, key])
                cands.append([new_c])
    return cands

def run_embeddings_system(data, embeddings, iteration, naf_folder, nl_nlp, graph_filename):
    """
    Run the embeddings system.
    """
    refined_news_items=copy.deepcopy(data)
    print('Now pre-grouping clusters...')
    candidate_clusters=pregroup_clusters(refined_news_items)
    print('Clusters pre-grouped. We have %d groups.' % len(candidate_clusters))
    print(candidate_clusters)
    if config.sys_name=='string_features':
        new_ids={}
        for cid, members in enumerate(candidate_clusters):
            for mention, eid in members:
                new_ids[eid]=str(cid)
    else: # sys_name=='embeddings'
        #m2id=algorithm.construct_m2id(refined_news_items)
        #print('M2ID', len(m2id.keys()))
        new_ids=algorithm.cluster_identities(candidate_clusters, 
                                             embeddings,
                                             max_d=58)
    #assert len(new_ids)==len(ids), 'Mismatch between old and new ids. Old=%d; new=%d' % (len(ids), 
    #                                                                                    len(new_ids))
    refined_news_items=algorithm.replace_identities(refined_news_items,
                                                    new_ids)

    # ANALYZE
#    inspect_data(refined_news_items)

    naf_iter = naf_folder / str(iteration)
    naf.create_nafs(naf_iter, 
                    refined_news_items, 
                    nl_nlp)

    naf.add_ext_references_to_naf(refined_news_items,
                               f'iteration{iteration}',
                               naf_folder / str(iteration-1),
                               naf_iter)

    algorithm.generate_graph(refined_news_items, graph_filename)

    ids=analysis.inspect_data(refined_news_items, graph_filename)
    
    return refined_news_items, new_ids

if __name__=="__main__":

    # LOAD CONFIG DATA
    bert_model=config.bert_model
    prefix=config.uri_prefix
    entity_layer=config.naf_entity_layer
    ner_system=config.ner

    system_name=config.sys_name  
  
    data_dir=Path(config.data_dir)
    input_dir=Path(config.input_dir)
    sys_dir=Path(config.sys_dir)
   
    if not os.path.exists(data_dir):
        data_dir.mkdir() 

    this_sys_dir=Path('%s/%s' % (sys_dir, system_name))
    if not os.path.exists(this_sys_dir):
        this_sys_dir.mkdir()

    naf_dir=Path('%s/naf' % this_sys_dir)
    el_file=Path('%s/el.pkl' % this_sys_dir)
    graphs_file=Path('%s/graphs.graph' % this_sys_dir)

    if os.path.exists(naf_dir):
        shutil.rmtree(str(naf_dir))
    naf_dir.mkdir()

    emb_dir=Path('%s/emb' % data_dir)
    if not os.path.exists(emb_dir):
        emb_dir.mkdir()

    print('Directories refreshed.')

    # LOAD MODELS
    
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained(bert_model, 
                                              do_lower_case=False)
    print('BERT tokenizer loaded')

    # Load pre-trained model (weights)
    model = BertModel.from_pretrained(bert_model)
    print('BERT model loaded')
    
    nl_nlp=nl_core_news_sm.load()
    
    doc2vec_model=doc2vec.get_doc2vec_model(str(emb_dir), str(input_dir), force=False)
    print(doc2vec_model.docvecs[0].shape)
    print('Doc2Vec model loaded')

    # ------ Generate NAFs and fill classes with entity mentions (Steps 1 and 2) --------------------

    #TODO: COMBINE NAFs with classes processing to run spacy only once!
    news_items_with_entities=load_utils.get_docs_with_entities(str(data_dir),
                                                               str(input_dir),
                                                               nl_nlp,
                                                               ner_system)

    naf_empty=naf_dir / 'empty'
    naf.create_nafs(naf_empty, news_items_with_entities, nl_nlp, ner_system)

    naf0=naf_dir / '0' # NAF folder before iteration 1
    if ner_system=='gold':
        naf.add_ext_references_to_naf(news_items_with_entities,
				       'gold',
				       naf_empty,
				       naf0)

    # ------- Run embeddings system -----------------

    iteration=1
    # BERT embeddings
    entity_embeddings, sent_embeddings, all_emb=emb_utils.get_entity_and_sentence_embeddings(naf_dir, 
                                                                                  iteration, 
                                                                                  model, 
                                                                                  tokenizer,
                                                                                  news_items_with_entities,
                                                                                  modify_entities=config.modify_entities)
    print('Entity and sentence embeddings created')
    print(sent_embeddings['Leipzig'].keys())
    print(entity_embeddings['Leipzig'].keys())
    print(all_emb['Leipzig'].keys())
    print(sent_embeddings['Leipzig']['1'].shape)
    print(entity_embeddings['Leipzig']['e1'].shape)
    print(all_emb['Leipzig']['e1'].shape)
    #id_embeddings=emb_utils.sent_to_id_embeddings(sent_embeddings, 
    #                                              data)

    full_embeddings=defaultdict(dict)
    for i, item in enumerate(news_items_with_entities):
        doc_vector=doc2vec_model.docvecs[i]
        current_emb=all_emb[item.identifier]
        for eid, embs in current_emb.items():
            full_embeddings[item.identifier][eid]=np.concatenate((embs, doc_vector), axis=0)

    print('Full embedding shape', full_embeddings['Leipzig']['e2'].shape)
    data, ids = run_embeddings_system(news_items_with_entities, 
                                    full_embeddings, 
                                    iteration, 
                                    naf_dir,
                                    nl_nlp,
                                    graphs_file)
