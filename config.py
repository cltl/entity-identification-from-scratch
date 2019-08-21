# ----- Data settings ----- #
max_documents=None #200 # for debugging; otherwise set it to None
#corpus_name='wikinews'
corpus_name='dbpedia_abstracts'

# ----- Baselines settings ----- #
factors=['docid', 'type']

# ----- Embeddings settings ----- #
bert_model='bert-base-multilingual-cased'
#sys_name='string_features' # 
sys_name='embeddings'
modify_entities=False

# ----- Other settings ----- #
ner='gold' # gold or spacy
uri_prefix='http://cltl.nl/entity#'
naf_entity_layer='entities'


# ------ INFERRED ------ #
if max_documents:
    corpus_name+=str(max_documents)

# Generate directory names
data_dir='data/%s' % corpus_name

raw_input_dir='%s/input_data' % data_dir
input_dir='%s/documents' % data_dir
sys_dir='%s/system' % data_dir

# Corpus URI
corpus_uri='http://%s.nl' % corpus_name
