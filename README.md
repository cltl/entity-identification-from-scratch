# entity-detection-for-historical-dutch

Entity recognition and linking for historical documents in Dutch, developed within the Clariah+ project

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

### Code structure

* The script `extract_corpus.py` creates a first corpus we will work with. It expects the file `data/input_data/nlwikinews-latest-pages-articles.xml` as input, which is a collection of Wikinews documents in Dutch in XML format. This script uses some functions from `load_utils.py`.

* The script `main.py` executes the procedure described above. It relies on the utility file `entity_utils.py`, on the classes file `classes.py`, and on the configuration file `config.py`.

* The files `inspect_embeddings.py` and `Inspect_data.ipynb` are currently used for debugging and will be removed in the near future.

### Next steps

* Debugging:
    * words not in vocabulary
    * fix identity links in the data loading procedure
    * improve the data loading, i.e., pay attention to templates
* Inspect MISC entities
* Evaluation by using the gold links extracted
* Employ Doc2Vec