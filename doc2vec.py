import os
import pickle
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import statistics

import pickle_utils as load

# ------ Processing news items -------------------

def load_docs_for_evaluation(bindir, docid, ids=None):
    """Loads news items, cleans them and associates them to their identifier index"""
    if ids is None:
        with open(bindir + "/doc2vec.ids", 'rb') as f:
            ids = pickle.load(f)
    docs = []
    for doc in load.load_news_items(bindir, docid):
        docs.append(TaggedDocument(clean_wiki(doc.content), [ids.index(doc.identifier)]))
    return docs

def load_docs_for_training(embdir, datadir):
    """Cleans wiki news items and formats them as TaggedDocument for Doc2Vec"""

    ids = []
    docs = []
    pkl_docs='%s.pkl' % datadir
    for i, doc in enumerate(load.load_news_items(pkl_docs)):
        ids.append(doc.identifier)
        docs.append(TaggedDocument(doc.content, [i]))
    with open(embdir + "/doc2vec.ids", 'wb') as w:
        pickle.dump(ids, w)
    return docs

# ------- Loading / training doc2vec model ----------

def run_doc2vec(docs, model_dir, size=1000):
    """Trains doc2vec model

    code source: https://rare-technologies.com/doc2vec-tutorial/
    """
    model = Doc2Vec(size=size, alpha=0.025, min_alpha=0.025)  # use fixed learning rate
    model.build_vocab(docs)
    for epoch in range(10):
        model.train(docs, total_examples=model.corpus_count, epochs=model.epochs)
        model.alpha -= 0.002  # decrease the learning rate
        model.min_alpha = model.alpha  # fix the learning rate, no decay
    model.save(model_dir + "/doc2vec.model")
    return model


def get_doc2vec_model(model_dir, input_dir, force=False, size=1000):
    """Loads or trains model

    Forces retraining if force == True"""
    if os.path.isfile(model_dir + "/doc2vec.model") and not force:
        print("loading Doc2vec model from {}".format(model_dir))
        return Doc2Vec.load(model_dir + "/doc2vec.model")
    else:
        print("training Doc2vec model")
        docs = load_docs_for_training(model_dir, input_dir)
        return run_doc2vec(docs, model_dir, size=size)


# --------- model evaluation --------------------


def first_10_words(docs, i):
    """Get the first 10 words of a document"""
    return ' '.join(docs[i][0][:10])

def get_index(docs, id):
    """Finds the index of the first doc with a matching id

    defaults to -1"""
    return next((i for i, d in enumerate(docs) if d[1] == id), -1)


def most_similar_vecs(docs, i, model, topn=10):
    """Obtain the most similar N vectors for a vector"""
    new_vector = model.infer_vector(docs[i][0])
    return model.docvecs.most_similar([new_vector], topn=topn)


def compare_to_most_similar_doc(docs, i, model):
    """Prints beginning of document i and of its most similar document"""
    sims = most_similar_vecs(docs, i, model, topn=1)
    print("doc: {}...".format(first_10_words(docs, i)))
    print("most similar: {}...".format(first_10_words(docs, get_index(docs, sims[0][0]))))


def rank(i, docs, model, topn=10):
    """Get the rank of document i in model's most similar documents"""
    sims = most_similar_vecs(docs, i, model, topn=topn)
    r = -1
    for s in sims:
        if s[0] == docs[i][1]:
            return sims.index(s)
    return r


def rank_to_self(docs, model, nb_samples=100, topn=10):
    """Self ranking statistics

    Number of documents that appear in top 10 of most similar documents to themselves,
    and mean and standard deviation of self rank in top 10
    """
    if nb_samples is None:
        nb_samples = len(docs)
    ranks = [rank(i, docs, model, topn=topn) for i in range(nb_samples)]
    topn_ranks = [r for r in ranks if r != -1]
    within_topn_ratio = len(topn_ranks) * 1.0 / nb_samples
    top1_ratio = len([r for r in ranks if r == 0]) * 1.0 / nb_samples
    return statistics.mean(topn_ranks), statistics.stdev(topn_ranks), within_topn_ratio, top1_ratio, nb_samples

def evaluate(model, docs, nb_samples=100, topn=10):
    """Evaluate the model in terms of similarity of documents."""
    mean, stdev, topn_ratio, top1_ratio, nb_samples = rank_to_self(docs, model, nb_samples=nb_samples, topn=topn)
    print("similarity to self ({} samples)".format(nb_samples))
    print("within top-{} ratio = {}; top1 ratio = {}; mean = {}; stdev = {}".format(topn, topn_ratio, top1_ratio, mean, stdev))

    for i in range(5):
        print()
        compare_to_most_similar_doc(docs, i, model)


# ---------- Main ----------------------


if __name__ == "__main__":
    doc_name = 'documents'
    bin_dir = 'bin'

    doc2vec_model = get_doc2vec_model(bin_dir, doc_name, force=False)
    evaluate(doc2vec_model, load_docs_for_evaluation(bin_dir, doc_name), nb_samples=None, topn=250)
