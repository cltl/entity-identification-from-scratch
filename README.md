# entity-detection-for-historical-dutch

Entity recognition and linking for historical documents in Dutch, developed within the Clariah+ project at VU Amsterdam

The current solution is entirely unsupervised, and works as follows:
1. Obtain documents
2. Extract entity mentions (SpaCy)
3. Assume entity identity criterion (e.g., one-form-one-entity)
4. Generate baseline graph
5. Generate entity and word embeddings
6. Refine graph identities by clustering on top of embedding similarity
7. Retrain embeddings

\*Repeat step 6 and 7 until convergence
\*Repeat steps 4-7 for different assumptions in step 3

### Authors

* Filip Ilievski (f.ilievski@vu.nl)

### Code structure

* The script `make_wiki_corpus.py` creates a first corpus we will work with, in JSON format and as Python classes. It expects the file `data/input_data/nlwikinews-latest-pages-articles.xml` as input, which is a collection of Wikinews documents in Dutch in XML format. This script uses some functions from `load_utils.py`.

* The script `main.py` executes the procedure described above. It relies on the utility file `entity_utils.py`, on the classes file `classes.py`, and on the configuration file `config.py`.

* The files `inspect_embeddings.py` and `Inspect_data.ipynb` are currently used for debugging and will be removed in the near future.

### Next steps

* [must] Create NAF at the initial document processing, add layers during the refinements
    * Refactoring: run SpaCy only once - connect NAF and classes
* [must] Ensure the circular approach works (for iteration >=2) until convergence
    * ensure iteration>=2 works
* [analysis] Evaluate the accuracy of the approach by using the [incomplete] links we have
    * For each of the 4 baseline assumptions
    * At various iterations of refinement
* [enhancement] initialize word2vec with pre-trained word embeddings (and zeros for the rest)
* [enhancement] Incorporate doc2vec
* [enhancement] implement other identity assumptions 
    * based on events/incidents 
    * based on document similarity
* [general] Try a different dataset
