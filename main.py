import pickle
import os.path
import copy
from gensim.models import Word2Vec
import networkx as nx
from collections import defaultdict
import nl_core_news_sm
import sys
from path import Path
import shutil
import glob

import load_utils
import entity_utils as utils
import config

nl_nlp=nl_core_news_sm.load()

def generate_identity(objs, 
                      prefix='http://cltl.nl/entity#', 
                      factors=[],
                      filename=''):
    """
    Decide which entities are identical, based on a set of recognized entity mentions and flexible set of factors.
    """
    data=copy.deepcopy(objs)
    for news_item in data:
        for mention in news_item.sys_entity_mentions:
            mention.identity=utils.strip_identity(mention.mention)
            #'%s%s' % (prefix, mention.mention)
            if 'docid' in factors:
                mention.identity+=news_item.identifier.split('_')[-1]
            if 'type' in factors:
                mention.identity+=mention.the_type

    with open(filename, 'wb') as w:
        pickle.dump(data, w)

    utils.generate_graph(data, filename.replace('/el', '/graphs'))

    return data

def generate_embeddings(news_items_with_entities, save_loc=''):
    """Generate embeddings based on a collection of news items with entity identity."""
    if save_loc and os.path.isfile(save_loc):
        model = Word2Vec.load(save_loc)
        return model
    all_sentences=utils.load_sentences(nl_nlp, news_items_with_entities)
    model = Word2Vec(all_sentences,
                     min_count=1,   # Ignore words that appear less than this
                     size=200,      # Dimensionality of word embeddings
                     workers=2,     # Number of processors (parallelisation)
                     window=5,      # Context window for words during training
                     iter=30)       # Number of epochs training over corpus
    if save_loc:
        model.save(save_loc)
    return model
    
def inspect_data(data, graph_file):
    """Analyze basic properties of the data."""
    with_types='type' in graph_file
    g=None
    if with_types:
        g=nx.read_gpickle(graph_file)
    utils.inspect(data, with_types, g)

def embeddings_in_a_doc(embeddings, d):
    for e in embeddings:
        if d in e:
            print(e)

def identity_vs_embeddings_stats(data, embeddings):
    stats=defaultdict(int)
    for news_item in data:
        for entity_mention in news_item.sys_entity_mentions:
            identity=utils.strip_identity(entity_mention.identity)
            #print(entity_mention.mention, entity_mention.identity, identity, identity in embeddings)
            stats[identity in embeddings]+=1
            if not identity in embeddings:
                print(identity)
    print(stats)

def get_docs_with_entities(bindir, input_dir):
    """Obtain news items processed with NER."""
    pkl_docs='%s/%s.pkl' % (bindir, input_dir)
    ent_addon='_with_ent'
    pkl_docs_with_entities='%s/%s%s.pkl' % (bindir, input_dir, ent_addon)
    
    if os.path.isfile(pkl_docs_with_entities):
        print('pickle file with recognized entities exists. Loading it now...')
        news_items_with_entities=load_utils.load_news_items(bindir, 
                                                            input_dir + ent_addon)
    else:
        print('Pickle file does not exist. Let us load the news items and run NER...')
        news_items=load_utils.load_news_items(bindir, input_dir)
        print('Loaded %d news items' % len(news_items))
        news_items_with_entities=utils.recognize_entities(nl_nlp, news_items)
        load_utils.save_news_items(bindir, 
                                   input_dir + ent_addon, 
                                   news_items_with_entities)
    return news_items_with_entities

def create_nafs(naf_folder, 
                news_items, 
                layers={'raw', 'text', 'terms', 'entities'}):
    """Create NAFs if not there already."""
    
    if naf_folder.exists():
        file_count =  len(glob.glob('%s/*.naf' % naf_folder))
        assert file_count == len(news_items), 'NAF directory exists, but a wrong amount of files. Did you edit the source documents?'
        print('NAF files were already there, and at the right number.')
        #shutil.rmtree(str(naf_folder))
    else:
        print('No NAF files found. Let\'s create them.')
        naf_folder.mkdir()
        layers={'raw', 'text', 'terms', 'entities'}
        utils.create_naf_for_documents(news_items, 
                                       layers, 
                                       nl_nlp, 
                                       naf_folder)

def patch_classes_with_tokens(news_items, naf_dir, entity_layer):
    for item in news_items:
        docid=item.identifier
        naf_output_path = naf_dir / f'{docid}.naf'
        eid_to_tids=utils.obtain_entity_data(naf_output_path, entity_layer)
        for e in item.sys_entity_mentions:
            eid=e.eid
            e.tokens=eid_to_tids[eid]
    return news_items
        
if __name__ == "__main__":

    bindir=Path('bin')
    input_dir=Path('documents')
    naf_folder = bindir / 'naf'
    naf0=naf_folder / '0'
    # ------ Generate NAFs and fill classes with entity mentions (Steps 1 and 2) --------------------
    
    #TODO: COMBINE NAFs with classes processing to run spacy only once!
    news_items_with_entities=get_docs_with_entities(str(bindir), 
                                                    str(input_dir))
    create_nafs(naf0, news_items_with_entities)

    entity_layer='entities'
    #news_items_with_entities=patch_classes_with_tokens(news_items_with_entities,
    #                                                   naf0, 
    #                                                   entity_layer)
        
    # Generate baseline graphs
    all_factors=config.factors
    for factor_combo in utils.get_variable_len_combinations(all_factors):
        iteration=1
        # ------ Pick identity assumption (Step 3) --------------------
        
        if len(factor_combo)<2: continue
        print('Assuming identity factors:', factor_combo)
        
        # ------ Run iteration 1 --------------------

        # GENERATE IDENTITIES (Step 4)
        print('Generating identities and graphs...')
        
        el_file='bin/el/mention_%s_graph.pkl' % '_'.join(factor_combo)
        data=generate_identity(news_items_with_entities, 
                               factors=factor_combo, 
                               filename=el_file)
        
        utils.add_ext_references(data, 
                           f'iteration{iteration}', 
                           naf0, 
                           naf_folder / str(iteration))
        sys.exit()
        # ANALYZE IDENTITIES
        graph_file='bin/graphs/mention_%s_graph.graph' % '_'.join(factor_combo)
        inspect_data(data, graph_file)

        # GENERATE (OR LOAD) EMBEDDINGS (Step 5)
        emb_file='bin/emb/emb_%s.model' % '_'.join(factor_combo)
        print('done.')
        print('Generating initial embeddings...')
        embeddings=generate_embeddings(data, 
                                       save_loc=emb_file)
    
        # ANALYZE EMBEDDINGS
        #embeddings_in_a_doc(embeddings.wv.vocab, '3382')
        identity_vs_embeddings_stats(data, embeddings.wv.vocab)
        old_len_vocab=len(embeddings.wv.vocab)
        print('Iteration 1 finished! Now refining...')
                
        sys.exit()
        while True:
            iteration=2
            print()
            print('Starting ITERATION:', iteration)
            print()
            # TODO: FIX THIS PART!
            refined_news_items=copy.deepcopy(data)
            m2id=utils.construct_m2id(refined_news_items)
            new_ids=utils.cluster_identities(m2id, embeddings.wv)

            refined_news_items=utils.replace_identities(refined_news_items, 
                                                        new_ids)
            
            # ANALYZE
            inspect_data(refined_news_items)
            
            # GENERATE EMBEDDINGS
            print('done.')
            print('Generating initial embeddings...')
            embeddings=generate_embeddings(refined_news_items)
            
            print(len(wv.vocab))
            if old_len_vocab==len(embeddings.wv.vocab):
                print('No change in this iteration. Done refining...')
                break
            
            old_len_vocab=len(embeddings.wv.vocab)
            
            iteration+=1
        
            data=refined_news_items
        
            break
