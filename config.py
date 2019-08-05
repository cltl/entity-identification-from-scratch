factors=['docid', 'type']
bert_model='bert-base-multilingual-cased'

uri_prefix='http://cltl.nl/entity#'

naf_entity_layer='entities'

corpus_name='wikinews200'
corpus_uri='http://wikinews.nl'
max_documents=200 # for debugging; otherwise set it to None

# Generate directory names
data_dir='data/%s' % corpus_name

raw_input_dir='%s/input_data' % data_dir
input_dir='%s/documents' % data_dir
naf_dir='%s/naf' % data_dir
el_dir='%s/el' % data_dir
graphs_dir='%s/graphs' % data_dir
