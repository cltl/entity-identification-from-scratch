import pickle
import pprint
import entity_utils as utils
pp = pprint.PrettyPrinter(indent=4)

pickle_file='bin/el/mention_docid_type_graph.pkl'
with open(pickle_file, 'rb') as f:
    news_items_with_entities=pickle.load(f)

for n in news_items_with_entities:
    if '1573' not in n.identifier: continue
    for em in n.sys_entity_mentions:
        print(em.mention, em.identity)

print(utils.load_sentences(news_items_with_entities))
