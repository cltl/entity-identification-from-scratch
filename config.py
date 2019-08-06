factors=['docid', 'type']
bert_model='bert-base-multilingual-cased'
ner='gold' # gold or spacy

uri_prefix='http://cltl.nl/entity#'

naf_entity_layer='entities'

max_documents=200 # for debugging; otherwise set it to None
corpus_name='wikinews'
#corpus_name='dbpedia_abstracts'


# ------ INFERRED -------
if max_documents:
    corpus_name+=str(max_documents)

# Generate directory names
data_dir='data/%s' % corpus_name

raw_input_dir='%s/input_data' % data_dir
input_dir='%s/documents' % data_dir
naf_dir='%s/naf' % data_dir
el_dir='%s/el' % data_dir
graphs_dir='%s/graphs' % data_dir

corpus_uri='http://%s.nl' % corpus_name
