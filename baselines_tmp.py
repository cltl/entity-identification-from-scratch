import pathlib
import shutil
import copy
import nl_core_news_sm
from path import Path
import os.path
import pickle_utils as pkl
import analysis_utils as analysis
import algorithm_utils as algorithm
import naf_utils as naf
import config
import wip.naf_handler as nafh


def run_baseline(factor_combo, news_items_with_entities, naf_folder, el_file, graphs_file, prefix):
    """Run an identity baseline."""
    iteration = 1
    # GENERATE IDENTITIES (Step 4)
    print('Generating identities and graphs...')

    data = algorithm.generate_identity(news_items_with_entities,
                                       factors=factor_combo,
                                       prefix=prefix,
                                       el_filename=el_file,
                                       graph_filename=graphs_file)

    # ANALYZE IDENTITIES
    ids = analysis.inspect_data(data, graphs_file)

    nafh.add_ext_references(data, f'{naf_folder}/0', f'{naf_folder}/1')
    return data, ids


if __name__ == "__main__":


    cfg = config.create('cfg/abstracts_nif35.yml')
    # LOAD CONFIG DATA
    all_factors = cfg.factors  # all factors that we will use to distinguish identity in our baseline graphs
    prefix = cfg.uri_prefix
    entity_layer = cfg.naf_entity_layer
    ner_system = cfg.ner

    data_dir = Path(cfg.data_dir)
    input_dir = Path(cfg.input_dir)
    sys_dir = Path(cfg.sys_dir)

    if not os.path.exists(data_dir):
        data_dir.mkdir()

    nl_nlp = nl_core_news_sm.load()

    # ------ Generate NAFs and fill classes with entity mentions (Steps 1 and 2) --------------------

    # TODO: COMBINE NAFs with classes processing to run spacy only once!
    news_data = pkl.get_docs_with_entities(str(data_dir),
                                           str(input_dir),
                                           nl_nlp,
                                           ner_system)

    for factor_combo in algorithm.get_variable_len_combinations(all_factors):
        # Generate baseline graphs

        news_items_with_entities = copy.deepcopy(news_data)

        baseline_name_parts = ['baseline'] + list(factor_combo)
        baseline_name = '_'.join(baseline_name_parts)

        naf_dir = Path('%s/%s/naf' % (sys_dir, baseline_name))
        el_file = Path('%s/%s/el.pkl' % (sys_dir, baseline_name))
        graphs_file = Path('%s/%s/graphs.graph' % (sys_dir, baseline_name))

        if os.path.exists(naf_dir):
            shutil.rmtree(str(naf_dir))
        pathlib.Path(naf_dir).mkdir(parents=True, exist_ok=True)

        naf_empty = naf_dir / 'empty'
        naf.create_nafs(naf_empty, news_items_with_entities, nl_nlp, ner_system)

        naf0 = naf_dir / '0'  # NAF folder before iteration 1
        if ner_system == 'gold':
            nafh.add_ext_references_gold(news_items_with_entities, naf_empty, naf0)
            print('Gold links added')

        # ------ Pick identity assumption (Step 3) --------------------

        print('Assuming identity factors:', factor_combo)

        data, ids = run_baseline(factor_combo,
                                 news_items_with_entities,
                                 naf_dir,
                                 el_file,
                                 graphs_file,
                                 prefix)
        print('Iteration 1 finished for baseline', baseline_name)
