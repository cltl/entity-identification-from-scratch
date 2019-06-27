from lxml import etree
import json
import os
import glob

tree = etree.parse("data/input_data/nlwikinews-latest-pages-articles.xml")
root = tree.getroot()

for f in glob.glob('data/documents/*'):
    os.remove(f)

counter=1
for x in root.findall('{http://www.mediawiki.org/xml/export-0.10/}page'):
    title=x.find('{http://www.mediawiki.org/xml/export-0.10/}title').text
    if title.startswith('H:') or title.startswith('WN:') or title.startswith('Sjabloon:') or title.startswith('Bestand:') or title.startswith('Help:') or title.startswith('Module:') or title.startswith('Categorie:') or title.startswith('MediaWiki:') or title.startswith('Wikinieuws:') or 'Nieuwsbrief Wikimedia Nederland' in title:
        continue
    text=x.find('{http://www.mediawiki.org/xml/export-0.10/}revision/{http://www.mediawiki.org/xml/export-0.10/}text').text
    if len(text)<100:
        continue

    j={'title': title, 
       'body': text}    

    with open('data/documents/wiki_%d.json' % counter, 'w') as outfile:
        json.dump(j, outfile)
    counter+=1
