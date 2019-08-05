import pickle
import pprint
pp = pprint.PrettyPrinter(indent=4)


pickle_file='bin/el/mention_type_graph.pkl'
with open(pickle_file, 'rb') as f:
    news_items_with_entities=pickle.load(f)
ids=set()
for n in news_items_with_entities:
    for em in n.sys_entity_mentions:
        i=em.identity
        if 'MISC' in i:
            ids.add(i)
pp.pprint(ids)
