# entity-detection-for-historical-dutch

Entity identification for historical documents in Dutch, developed within the Clariah+ project at VU Amsterdam.

While the primary use case is to process historical Dutch documents, the more general idea of this project is to develop an adaptive framework that can process any set of Dutch documents. This means, for instance, documents with or without recognized entities (gold NER or not); documents with entities that can be linked or not (in-KB entities or not), etc.

We achieve this flexibility in two ways: 
1. we create an unsupervised system, based on recent techniques, like BERT embeddings
2. we involve human experts, by allowing them to enrich or alter the tool output

### Current algorithm in a nutshell

The current solution is entirely unsupervised, and works as follows:
1. Obtain documents (supported formats so far: mediawiki and NIF
2. Extract entity mentions (gold NER or by running SpaCy)
3. Create initial NAF documents containing recognized entities too
4. Compute BERT sentence+mention embeddings
5. Enrich them with word2vec document embeddings
6. Bucket mentions based on similarity of mentions
7. Cluster embeddings for a bucket based on the HAC algorithm
8. Run evaluation with rand index-based score

### Baselines

We compare our identity clustering algorithm against 5 baselines:
1. string-similarity - forms that are identical or sufficiently similar are coreferential.
2. one-form-one-identity - all occurrences of the same form refer to the same entity.
3. one-form-and-type-one-identity - all occurrences of the same form, when this form is of the same semantic type, refer to the same entity.
4. one-form-in-document-one-identity - all occurrences of the same form within a document are coreferential. All occurrences across documents are not.
5. one-form-and-type-in-document-one-identity - all occurrences of the same form that have the same semantic type within a document are coreferential; the rest are not.

### Code structure

* The scripts `make_wiki_corpus.py` and `make_nif_corpus.py` create a corpus (as Python classes) from the source data we download in mediawiki or NIF format, respectively. The script `make_wiki_corpus.py` expects the file `data/input_data/nlwikinews-latest-pages-articles.xml` as input, which is a collection of Wikinews documents in Dutch in XML format. The script `make_nif_corpus.py` expects the iput file `abstracts_nl{num}.ttl`, where `num` is a number between 0 and 43, inclusive. These extraction scripts use some functions from `pickle_utils.py` and from `wiki_utils.py`.

* The script `main.py` executes the algorithm procedure described above. It relies on functions in several utility files: `algorithm_utils.py`, `bert_utils.py`, `analysis_utils.py`, `pickle_utils.py`, `naf_utils.py`.

* Evaluation functions are stored in the file `evaluation.py`.

* Baselines are run by running the file `baselines.py` (with no arguments).

* The classes we work with are defined in the file `classes.py`.

* Configuration files are found in the folder `cfg`. These are loaded through the script `config.py`.

* All data is stored in the folder `data`.

### Preparation to run the script: Install and download

To prepare your environment with the right packages, run `bash install.sh`.

Then download the corpora you would like to work with, and store it in: `data/{corpus_name}/input_data`. To reuse the config files found in `cfg` and run wikinews or dbpedia abstracts, you can do the following.
1. `for wikinews`, download `nlwikinews-latest-pages-articles.xml`, for example from [here](https://archive.org/details/incr-nlwikinews-20190902). Then store it in `data/wikinews/input_data` (make sure you unpack it).
2. for `dbpedia_abstracts`, you can download .ttl files from [this link](http://downloads.dbpedia.org/2015-04/ext/nlp/abstracts/nl/). Each .ttl contains many abstracts, so it is advisable to start with 1 file to understand what is going on. Download and unpack the file, then store it in `data/dbpedia_abstracts/input_data`

Then you should be able to run `make_wiki_corpus.py` and `make_nif_corpus.py` to load the corpora; and you should be able to run directly `main.py` in order to process the corpora with our tool. Make sure that you use the right config file in these scripts (e.g., `wikinews50.cfg` will let you process 50 files from Wikinews).

### Authors

* Filip Ilievski (f.ilievski@vu.nl)
* Sophie Arnoult (sophie.arnoult@posteo.net)
