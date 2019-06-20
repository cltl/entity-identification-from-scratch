import pickle
import os.path
import copy
from gensim.models import Word2Vec

import load_utils
import entity_utils as utils
import config


def generate_identity(objs, 
                      prefix='http://cltl.nl/entity#', 
                      factors=[]):
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

    filename='bin/el/mention_%s_graph.pkl' % '_'.join(factors)

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
                     min_count=2,   # Ignore words that appear less than this
                     size=200,      # Dimensionality of word embeddings
                     workers=2,     # Number of processors (parallelisation)
                     window=5,      # Context window for words during training
                     iter=30)       # Number of epochs training over corpus
    if save_loc:
        model.save(save_loc)
    return model
    
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
        print('Assuming identity factors:', factor_combo)
        print('Generating identities and graphs...')
        data=generate_identity(news_items_with_entities, 
                          factors=factor_combo)
        emb_file='bin/emb/emb_%s.model' % '_'.join(factor_combo)
        print('done.')
        print('Generating initial embeddings...')
        embeddings=generate_embeddings(data, 
                                       save_loc=emb_file)
        print('DONE!')
        break