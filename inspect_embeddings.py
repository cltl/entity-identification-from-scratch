import entity_utils as utils

from gensim.models import Word2Vec
import pickle
import numpy as np
import glob
import spacy
from collections import defaultdict
import nl_core_news_sm
nl_nlp=nl_core_news_sm.load()
            
if __name__ == "__main__":
    model = Word2Vec.load('bin/emb/emb_docid_type.model')
    
    
    f='bin/el/mention_docid_type_graph.pkl'
    with open(f, 'rb') as pkl_file:
        print(f)
        news_items_with_entities=pickle.load(pkl_file)

        m2id=utils.construct_m2id(news_items_with_entities)
        new_ids=utils.cluster_identities(m2id, wv)

        news_items_with_entities=utils.replace_identities(news_items_with_entities,
                                                   new_ids)
        

          
