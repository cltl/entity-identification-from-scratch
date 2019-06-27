from gensim.models import Word2Vec
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np



if __name__ == "__main__":
    input_dir='documents'
    with open('../bin/%s.pkl' % input_dir, 'rb') as f:
        news_items_with_entities=pickle.load(f)


all_sentences=[]
for news_item in news_items_with_entities:
    import nl_core_news_sm
    nl_nlp = nl_core_news_sm.load()
    #print(news_item.title)
    text=f"{news_item.title}\n{news_item.content}"
    nl_doc=nl_nlp(text)
    for sent in nl_doc.sents:
        sent_tokens = [t.text for t in sent]
        all_sentences.append(sent_tokens)

print(len(all_sentences))

model = Word2Vec(all_sentences,
                 min_count=2,   # Ignore words that appear less than this
                 size=200,      # Dimensionality of word embeddings
                 workers=2,     # Number of processors (parallelisation)
                 window=5,      # Context window for words during training
                 iter=30)       # Number of epochs training over corpus


def compute_similarity(w1, w2, vectors):
    v1=np.array(vectors[w1]).reshape(1, -1)
    v2=np.array(vectors[w2]).reshape(1, -1)
    return cosine_similarity(v1, v2)


w1='Uitnodiging'
w2='welkom'
s=compute_similarity(w1, w2, model)
print(s)
