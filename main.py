import pickle
import os.path
import copy
from gensim.models import Word2Vec
import networkx as nx

import load_utils
import entity_utils as utils
import config


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
            mention.identity='%s%s' % (prefix, mention.mention)
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
    all_sentences=utils.load_sentences(news_items_with_entities)
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
    
if __name__ == "__main__":

    input_dir='documents'
    pickle_file='bin/%s.pkl' % input_dir
    if os.path.isfile(pickle_file):
        print('pickle file with recognized entities exists. Loading it now...')
        with open(pickle_file, 'rb') as f:
            news_items_with_entities=pickle.load(f)
    else:
        print('Pickle file does not exist. Let us load the news items and run NER...')
        news_items=load_utils.load_news_items(input_dir)
        news_items_with_entities=utils.recognize_entities(news_items)
        with open('bin/%s.pkl' % input_dir, 'wb') as w:
            pickle.dump(news_items_with_entities, w)

    # Generate baseline graphs
    all_factors=config.factors
    for factor_combo in utils.get_variable_len_combinations(all_factors):
        if len(factor_combo)<2: continue
        print('Assuming identity factors:', factor_combo)
        
        # GENERATE IDENTITIES
        print('Generating identities and graphs...')
        el_file='bin/el/mention_%s_graph.pkl' % '_'.join(factor_combo)
        data=generate_identity(news_items_with_entities, 
                          factors=factor_combo, filename=el_file)

        # ANALYZE
        graph_file='bin/graphs/mention_%s_graph.graph' % '_'.join(factor_combo)
        inspect_data(data, graph_file)
        
        # GENERATE (OR LOAD) EMBEDDINGS
        emb_file='bin/emb/emb_%s.model' % '_'.join(factor_combo)
        print('done.')
        print('Generating initial embeddings...')
        embeddings=generate_embeddings(data, 
                                       save_loc=emb_file)
        
        old_len_vocab=embeddings.wv.vocab
        print('DONE!')
        
        while True:
            iter=2
            print()
            print('ITERATION:', iter)
            print()
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
            
            iter+=1
        
            data=refined_news_items
        break