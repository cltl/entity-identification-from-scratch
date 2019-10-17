import load_utils
from config import Config

# Specify your config file here:
cfg = Config('cfg/abstracts50.yml')
cfg.setup_input()

# Load configuration variables
max_docs = cfg.max_documents
min_length=cfg.min_text_length


# Load a number of news items from a NIF file
news_items = load_utils.load_article_from_nif_file(cfg.raw_input, 
                                       limit=max_docs or 1000000,
                                       collection=cfg.corpus_name)

# Save the news articles to pickle
load_utils.save_news_items('%s/documents.pkl' % config.data_dir, news_items)
