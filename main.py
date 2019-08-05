import copy
import nl_core_news_sm
import sys
from path import Path
from pytorch_pretrained_bert import BertTokenizer, BertModel

import load_utils
import analysis_utils as analysis
import algorithm_utils as algorithm
import naf_utils as naf
import embeddings_utils as emb_utils
import config
    
################## Run iteration 1 or 2 or more #########################

def run_iteration_1(factor_combo, news_items_with_entities, naf_folder, el_dir, graphs_dir, prefix):
    iteration=1
    # GENERATE IDENTITIES (Step 4)
    print('Generating identities and graphs...')

    el_file='%s/mention_%s_graph.pkl' % (el_dir, '_'.join(factor_combo))
    graph_file='%s/mention_%s_graph.graph' % (graphs_dir, '_'.join(factor_combo))
    data=algorithm.generate_identity(news_items_with_entities,
                           factors=factor_combo,
                           prefix=prefix,
                           el_filename=el_file,
                           graph_filename=graph_file)

    # ANALYZE IDENTITIES
    ids=analysis.inspect_data(data, graph_file)
    
    naf.add_ext_references_to_naf(data,
                                   f'iteration{iteration}',
                                   naf_folder / str(iteration-1),
                                   naf_folder / str(iteration))

    return data, ids

def run_iteration_2_or_more(factor_combo, data, id_embeddings, ids, iteration, naf_folder, nl_nlp):
    refined_news_items=copy.deepcopy(data)
    m2id=algorithm.construct_m2id(refined_news_items)
    new_ids=algorithm.cluster_identities(m2id, 
                                         id_embeddings)
    assert len(new_ids)==len(ids), 'Mismatch between old and new ids. Old=%d; new=%d' % (len(ids), 
                                                                                         len(new_ids))
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
    print('done.')
    
    return refined_news_items, \
           set(new_ids.values())

if __name__=="__main__":

    # LOAD CONFIG DATA
    all_factors=config.factors # all factors that we will use to distinguish identity in our baseline graphs
    bert_model=config.bert_model
    prefix=config.uri_prefix
    entity_layer=config.naf_entity_layer
    
    data_dir=Path(config.data_dir)
    input_dir=Path(config.input_dir)
    naf_dir=Path(config.naf_dir)
    el_dir=Path(config.el_dir)
    graphs_dir=Path(config.graphs_dir)
        
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained(bert_model, 
                                              do_lower_case=False)
    print('BERT tokenizer loaded')

    # Load pre-trained model (weights)
    model = BertModel.from_pretrained(bert_model)
    print('BERT model loaded')
    
    nl_nlp=nl_core_news_sm.load()

    # ------ Generate NAFs and fill classes with entity mentions (Steps 1 and 2) --------------------

    #TODO: COMBINE NAFs with classes processing to run spacy only once!
    news_items_with_entities=load_utils.get_docs_with_entities(str(data_dir),
                                                               str(input_dir),
                                                               nl_nlp)
    
    naf0=naf_dir / '0' # NAF folder before iteration 1
    naf.create_nafs(naf0, news_items_with_entities, nl_nlp)

    #news_items_with_entities=patch_classes_with_tokens(news_items_with_entities,
    #                                                   naf0,
    #                                                   entity_layer)

    for factor_combo in algorithm.get_variable_len_combinations(all_factors):
        # Generate baseline graphs
        iteration=1
        # ------ Pick identity assumption (Step 3) --------------------

        if len(factor_combo)<2: continue
        print('Assuming identity factors:', factor_combo)

        # ------ Run iteration 1 --------------------
        print('Now running iteration 1')
        data, ids=run_iteration_1(factor_combo, 
                                  news_items_with_entities, 
                                  naf_dir, 
                                  el_dir, 
                                  graphs_dir, 
                                  prefix)
        print('Iteration 1 finished! Now refining...')

        # ------- Run iteration >=2 -----------------

        iteration=2
        old_len_vocab=0

        while True:
            print()
            print('Starting ITERATION:', iteration)
            print()

            # BERT embeddings
            sent_embeddings=emb_utils.get_sentence_embeddings(naf_dir, 
                                                              iteration, 
                                                              model, 
                                                              tokenizer)
            id_embeddings=emb_utils.sent_to_id_embeddings(sent_embeddings, 
                                                          data)

            data, ids = run_iteration_2_or_more(factor_combo, 
                                                data, 
                                                id_embeddings, 
                                                ids, 
                                                iteration, 
                                                naf_dir,
                                                nl_nlp)

            
            print('Entity embeddings in this iteration:', len(id_embeddings))
            if old_len_vocab==len(id_embeddings):
                print('No change in this iteration. Done refining...')
                break

            old_len_vocab=len(id_embeddings)

            iteration+=1
            break
