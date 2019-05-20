import classes

from pprint import pprint
import glob
import json
import pickle
import spacy
import nl_core_news_sm

nl_nlp=nl_core_news_sm.load()

def load_news_items(loc):
    news_items=set()
    for file in glob.glob('%s/*.json' % loc):
        with open(file, 'r') as f:
            data=json.load(f)
        news_item_obj=classes.NewsItem(
            content=data['body'],
            title=data['title'],
            identifier=file.split('.')[0],
            collection=loc
        )
        news_items.add(news_item_obj)
    return news_items

def recognize_entities(news_items):
    for i, news_item in enumerate(news_items):
        text=f"{news_item.title}\n{news_item.content}"
        nl_doc=nl_nlp(text)
        for e in nl_doc.ents:
            ent_mention_obj=classes.EntityMention(
                mention=e.text,
                begin_index=e.start,
                end_index=e.end,
                the_type=e.label_
            )
            news_item.entity_mentions.append(ent_mention_obj)
        print(i)
    return news_items

if __name__ == "__main__":

    input_dir='documents'
    news_items=load_news_items(input_dir)

    news_items_with_entities=recognize_entities(news_items)

    with open('bin/%s.pkl' % input_dir, 'wb') as w:
        pickle.dump(news_items_with_entities, w)
