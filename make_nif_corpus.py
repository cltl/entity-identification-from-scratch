import load_utils
import config

cfg = config.Config('cfg/nif_abstract_context_10.yml')
cfg.setup_input()

# raw_input_dir=config.raw_input_dir
# raw_input=raw_input_dir + '/leipzig_nl.ttl'
# max_docs=config.max_documents

news_items = load_utils.load_article_from_nif_files(cfg.raw_input_dir,
                                                    limit=cfg.max_documents or 1000000,
                                                    collection=cfg.corpus_name)

load_utils.save_news_items('%s/documents.pkl' % config.data_dir, news_items)
