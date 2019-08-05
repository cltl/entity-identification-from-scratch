import load_utils
import config

raw_input_dir=config.raw_input_dir
raw_input=raw_input_dir + '/leipzig_nl.ttl'
max_docs=config.max_documents

news_items = load_utils.load_article_from_nif_file(raw_input, 
                                       limit=max_docs,
                                       collection=config.corpus_name)

load_utils.save_news_items('%s/documents.pkl' % config.data_dir, news_items)
