from lxml import etree
import json
import os
import glob

import wiki_utils as utils
import pickle_utils as pkl
import classes
from config import Config

# Specify your config file here:
cfg = Config('cfg/wikinews50.yml')
cfg.setup_input()

# ------------------------------------------------------

# Loading of configuration variables
max_docs = cfg.max_documents
min_length=cfg.min_text_length

# Define some hard-coded areas of the files
marker_base='{http://www.mediawiki.org/xml/export-0.10/}'
title_marker=f'{marker_base}title'
page_marker=f'{marker_base}page'
text_marker=f'{marker_base}revision/{marker_base}text'

skip_starters=['H:', 'WN:', 'Sjabloon:', 'Bestand:', 'Help:', 'Module:', 'Categorie:', 'MediaWiki:', 'Wikinieuws:']
skip_contains=['Nieuwsbrief Wikimedia Nederland']

# Prepare variables and clean directories
news_items = set()

for f in glob.glob('%s/*.json' % cfg.input_dir):
    os.remove(f)

counter = 1

# Parse input XML file
tree = etree.parse(cfg.raw_input)
root = tree.getroot()

for x in root.findall(page_marker):
    title = x.find(title_marker).text
    # check is this a title we want to keep or skip
    for dont_start in skip_starters:
        if title.startswith(dont_start):
            continue
    for dont_contain in skip_contains:
        if dont_contain in title:
            continue

    # load text
    text = x.find(text_marker).text
    docid = 'wiki_%d' % counter

    # Create a news item object with the information from the text and the links    
    the_text, the_links = utils.get_text_and_links(text)
    clean_text = utils.clean_wiki(the_text)
    clean_title = utils.clean_wiki(title)
    if len(clean_text) < min_length:
        continue
    # TODO create_gold_mentions is deprecated (needs to be improved)
    news_item_obj = classes.NewsItem(
        content=clean_text,
        title=clean_title,
        identifier=docid,
        collection=cfg.corpus_name
        #gold_entity_mentions=utils.create_gold_mentions(the_links,
        #                                                clean_text)
    )
    news_items.add(news_item_obj)

    # Save it to JSON
    j = {'title': clean_title,
         'body': clean_text}
    with open('%s/%s.json' % (cfg.input_dir, docid), 'w') as outfile:
        json.dump(j, outfile)

    if max_docs and counter >= max_docs:
        break

    counter += 1

# Save the classes as pickle
pkl.save_news_items('%s/documents.pkl' % cfg.data_dir, news_items)
