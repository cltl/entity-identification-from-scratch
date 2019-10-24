import pathlib
import shutil
import copy
import sys

import nl_core_news_sm
from path import Path
import os.path
import pickle_utils as pkl
import analysis_utils as analysis
import algorithm_utils as algorithm
import naf_utils as naf
import config
import wip.naf_handler as nafh
import wip.embeddings as emb_utils


def run_baseline(factor_combo, news_items_with_entities, cfg):
    """Run an identity baseline."""
    iteration = 1
    # GENERATE IDENTITIES (Step 4)
    print('Generating identities and graphs...')

    data = algorithm.generate_identity(news_items_with_entities,
                                       factors=factor_combo,
                                       prefix=cfg.uri_prefix,
                                       el_filename=cfg.el_file_path(),
                                       graph_filename=cfg.graphs_file_path())

    # ANALYZE IDENTITIES
    ids = analysis.inspect_data(data, cfg.graphs_file_path())

    nafh.add_ext_references(data, cfg.this_naf_indir(), cfg.this_naf_outdir())
    return data, ids


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Missing config file argument. Now exiting.")
        exit(1)
    cfg = config.load(sys.argv[1])
    ner_system = cfg.ner

    # ------ Generate NAFs and fill classes with entity mentions (Steps 1 and 2) --------------------

    news_items = pkl.load_news_items(cfg.news_items_file())

    for factor_combo in algorithm.get_variable_len_combinations(cfg.factors):
        # Generate baseline graphs

        baseline_name_parts = ['baseline'] + list(factor_combo)
        baseline_name = '_'.join(baseline_name_parts)

        cfg.sys_name = baseline_name
        cfg.create_sysdirs()

        # 2. runs spacy and produces new NAF
        if cfg.create_input_nafs:
            nafh.run_spacy_and_write_to_naf(news_items, cfg.this_naf_indir())

        news_items_with_ner = nafh.load_news_items_with_entities(cfg.this_naf_indir())

        # ------ Pick identity assumption (Step 3) --------------------

        print('Assuming identity factors:', factor_combo)

        data, ids = run_baseline(factor_combo,
                                 news_items_with_ner,
                                 cfg)
        print('Processing finished for baseline', baseline_name)
