import pickle
import glob

def inspect(data):
    num_mentions=0
    identities=set()
    print(len(data), 'news documents')
    for news_item_obj in data:
        for m in news_item_obj.sys_entity_mentions:
            identities.add(m.identity)
        num_mentions+=len(news_item_obj.sys_entity_mentions)
    print('Num mentions', num_mentions)
    print('Num identities', len(identities))

input_dir='bin'
if __name__ == "__main__":
    for file in glob.glob('%s/*.pkl' % input_dir):
        print(file)
        with open(file, 'rb') as f:
            data=pickle.load(f)
        inspect(data)

