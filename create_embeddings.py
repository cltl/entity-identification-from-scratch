from gensim.models import Word2Vec
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import glob
import spacy
import nl_core_news_sm
nl_nlp=nl_core_news_sm.load()

def compute_similarity(w1, w2, vectors):
    """Compute similarity of 2 vectors."""
    v1=np.array(vectors[w1]).reshape(1, -1)
    v2=np.array(vectors[w2]).reshape(1, -1)
    return cosine_similarity(v1, v2)

if __name__ == "__main__":
    for f in glob.glob('bin/el/mention_*.pkl'):
        with open(f, 'rb') as pkl_file:
            print(f)
            news_items_with_entities=pickle.load(pkl_file)
            all_sentences=load_sentences(news_items_with_entities)
            model = Word2Vec(all_sentences,
                     min_count=2,   # Ignore words that appear less than this
                     size=200,      # Dimensionality of word embeddings
                     workers=2,     # Number of processors (parallelisation)
                     window=5,      # Context window for words during training
                     iter=30)       # Number of epochs training over corpus
            break
