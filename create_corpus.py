rom lxml import etree
import json

tree = etree.parse("input_data/nlwikinews-latest-pages-articles.xml")
root = tree.getroot()

counter=1
for x in root.findall('{http://www.mediawiki.org/xml/export-0.10/}page'):
    title=x.find('{http://www.mediawiki.org/xml/export-0.10/}title').text
    if title.startswith('Categorie:') or title.startswith('MediaWiki:') or title.startswith('Wikinieuws:') or 'Nieuwsbrief Wikimedia Nederland' in title:
        continue
    text=x.find('{http://www.mediawiki.org/xml/export-0.10/}revision/{http://www.mediawiki.org/xml/export-0.10/}text').text
    if len(text)<100:
        continue

    j={'title': title, 
       'body': text}    
    
    with open('documents/wiki_%d.json' % counter, 'w') as outfile:
        json.dump(j, outfile)
    counter+=1
