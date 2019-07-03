from lxml import etree
import json
import os
import glob

import load_utils as utils
import classes

tree = etree.parse("data/input_data/nlwikinews-latest-pages-articles.xml")
root = tree.getroot()

news_items=set()

for f in glob.glob('data/documents/*'):
    os.remove(f)

counter=1
for x in root.findall('{http://www.mediawiki.org/xml/export-0.10/}page'):
    title=x.find('{http://www.mediawiki.org/xml/export-0.10/}title').text
    if title.startswith('H:') or title.startswith('WN:') or title.startswith('Sjabloon:') or title.startswith('Bestand:') or title.startswith('Help:') or title.startswith('Module:') or title.startswith('Categorie:') or title.startswith('MediaWiki:') or title.startswith('Wikinieuws:') or 'Nieuwsbrief Wikimedia Nederland' in title:
        continue
    text=x.find('{http://www.mediawiki.org/xml/export-0.10/}revision/{http://www.mediawiki.org/xml/export-0.10/}text').text
        
    docid='wiki_%d' % counter
        
    
    # Create a news item object with the information from the text and the links    
    the_text, the_links=utils.get_text_and_links(text)
    clean_text=utils.clean_wiki(the_text)
    clean_title=utils.clean_wiki(title)
    if len(clean_text)<100:
        continue
    news_item_obj=classes.NewsItem(
            content=clean_text,
            title=clean_title,
            identifier=docid,
            collection='wikinews',
            gold_entity_mentions=utils.create_gold_mentions(the_links, 
                                                            clean_text)
    )
    news_items.add(news_item_obj)
    
    # Save it to JSON
    j={'title': clean_title, 
       'body': clean_text}    

    with open('data/documents/%s.json' % docid, 'w') as outfile:
        json.dump(j, outfile)
    counter+=1

# Save the classes
utils.save_news_items('bin', 'documents', news_items)