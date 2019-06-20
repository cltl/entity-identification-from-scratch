from gensim.models import Word2Vec
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import glob
import spacy
from collections import defaultdict
from sklearn.cluster import DBSCAN
import nl_core_news_sm
nl_nlp=nl_core_news_sm.load()

def compute_similarity(w1, w2, vectors):
    """Compute similarity of 2 vectors."""
    try:
        v1=np.array(vectors[w1]).reshape(1, -1)
    except:
        print(w1, 'not in vocab')
        return 0
    try:
        v2=np.array(vectors[w2]).reshape(1, -1)
    except:
        print(w2, 'not in vocab')
        return 0
    return cosine_similarity(v1, v2)

def cluster_matrix(distances, eps=0.1, min_samples=1):
    labels=DBSCAN(min_samples=min_samples, eps=eps, metric='precomputed').fit_predict(distances)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
        
    return list(labels), n_clusters, n_noise

def cluster_identities(m2id, wv):
    new_identities={}
    for m, ids in m2id.items():
        num_cands=len(ids)
        if num_cands<2: continue
        dist_matrix = np.zeros(shape=(num_cands, num_cands)) # Distances matrix
        ids=list(ids)
        for i, ent_i in enumerate(ids):
            for j, ent_j in enumerate(ids):
                if i>j:
                    dist=1-compute_similarity(ent_i, ent_j, wv)
                    print(ent_i, ent_j, dist)
                    dist_matrix[i,j]=dist
                    dist_matrix[j,i]=dist
        clusters, n_clusters, n_noise = cluster_matrix(dist_matrix, eps=0.4)
        print(m, ids, clusters)
        input('continue?')
        for index, cluster_id in enumerate(clusters):
            new_id='%s_%d' % (m, cluster_id)
            old_id=ids[index]
            new_identities[old_id]=new_id
    return new_identities

if __name__ == "__main__":
    model = Word2Vec.load('bin/emb/emb_docid_type.model')
    wv=model.wv
    
    m2id=defaultdict(set)
    
    f='bin/el/mention_docid_type_graph.pkl'
    with open(f, 'rb') as pkl_file:
        print(f)
        news_items_with_entities=pickle.load(pkl_file)
        for item in news_items_with_entities:
            for e in item.sys_entity_mentions:
                identity=e.identity.lstrip('http://cltl.nl/entity#').replace(' ', '_')
                if identity.endswith('MISC'): continue
                m2id[e.mention].add(identity)
        new_ids=cluster_identities(m2id, wv)

        for item in news_items_with_entities:
            for e in item.sys_entity_mentions:
                identity=e.identity.lstrip('http://cltl.nl/entity#').replace(' ', '_')
                new_identity=new_ids[identity]
                e.identity=new_identity
          
