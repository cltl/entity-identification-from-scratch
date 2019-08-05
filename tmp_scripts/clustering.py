sentences = ['hi', 'hello', 'hi hello', 'goodbye', 'bye', 'goodbye bye']
sentences_split = [s.lower().split(' ') for s in sentences]

import gensim
model = gensim.models.Word2Vec(sentences_split, min_count=2)

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage

l = linkage(model.wv.syn0, method='complete', metric='seuclidean')

max_d=14
clusters=fcluster(l, max_d, criterion='distance')
print(clusters)

print(model.wv.syn0)

# calculate full dendrogram
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.ylabel('word')
plt.xlabel('distance')

dendrogram(
    l,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=16.,  # font size for the x axis labels
    orientation='left',
    leaf_label_func=lambda v: str(model.wv.index2word[v])
)
plt.show()
