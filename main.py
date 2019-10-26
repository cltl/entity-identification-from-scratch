import sys

from path import Path

import Levenshtein as lev
import pickle_utils as pkl
import analysis_utils as analysis
import algorithm_utils as algorithm
import wip.embeddings as emb_utils
import config

import wip.naf_handler as nafh


################## Run iteration 1 or 2 or more #########################

def similar(a, b, tau=0.8):
    """
    Check if two strings are similar enough.
    """
    return lev.ratio(a, b) >= tau


def is_abbrev(abbrev, text):
    """Check if `abbrev` is a potential abbreviation of `text`."""
    abbrev = abbrev.lower()
    text = text.lower()
    words = text.split()
    if not abbrev:
        return True
    if abbrev and not text:
        return False
    if abbrev[0] != text[0]:
        return False
    elif words:
        return any(is_abbrev(abbrev[1:], text[i + 1:]) for i in range(len(words[0])))
    else:
        return False


def abbreviation(a, b):
    """Check if either a is an abbreviation of b, or vice versa."""
    return is_abbrev(a, b) or is_abbrev(b, a)


def pregroup_clusters(news_items_with_entities):
    """Pregroup entities into clusters based on string features."""
    cands = []
    for item in news_items_with_entities:
        for e in item.sys_entity_mentions:
            mention = e.mention
            key = '%s#%s' % (item.identifier, e.eid)
            found = False
            for c in cands:
                if found:
                    break
                for other_mention, other_identifier in c:
                    if similar(other_mention, mention) or abbreviation(other_mention, mention):
                        c.append(tuple([mention, key]))
                        found = True
                        break

            if not found:
                new_c = tuple([mention, key])
                cands.append([new_c])
    return cands


def run_embeddings_system(refined_news_items, embeddings, graph_filename, sys_name, cfg):
    """
    Run the embeddings system.
    """
    if not embeddings:
        raise ValueError('Full embeddings are empty. Refusing to continue')
    print('Now pre-grouping clusters...')
    candidate_clusters = pregroup_clusters(refined_news_items)
    print('Clusters pre-grouped. We have %d groups.' % len(candidate_clusters))
    print(candidate_clusters)
    if sys_name == 'string_features':
        new_ids = {}
        for cid, members in enumerate(candidate_clusters):
            for mention, eid in members:
                new_ids[eid] = str(cid)
    else:  # sys_name=='embeddings'
        new_ids = algorithm.cluster_identities(candidate_clusters,
                                               embeddings,
                                               max_d=58)
    refined_news_items = algorithm.replace_identities(refined_news_items,
                                                      new_ids)
    algorithm.generate_graph(refined_news_items, cfg.graphs_file_path())
    ids = analysis.inspect_data(refined_news_items, cfg.graphs_file_path())

    return refined_news_items, new_ids


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Missing config file argument. Now exiting. Usage:")
        print("python main.py {config_file}")
        exit(1)
    cfg = config.load(sys.argv[1])
    cfg.create_sysdirs()

    # ------ Generate NAFs and fill classes with entity mentions (Steps 1 and 2) --------------------

    # 1. news items loaded from pickle file;
    news_items = pkl.load_news_items(cfg.news_items_file())

    # 2. runs spacy and produces new NAF
    if cfg.create_input_nafs:
        nafh.run_spacy_and_write_to_naf(news_items, cfg.this_naf_indir())

    # ------- Run embeddings system -----------------

    # 3. creates entity embeddings
    iteration = 1
    tokenizer, model, d2v = emb_utils.load_models(cfg)
    full_embeddings, news_items_with_entities = emb_utils.get_entity_and_sentence_embeddings(
        cfg.this_naf_indir(),
        model,
        tokenizer,
        d2v)

    print('Entity and sentence embeddings created')

    # 4. clustering
    news_items_with_refs, ids = run_embeddings_system(news_items_with_entities, full_embeddings, cfg.graphs_file,
                                                        cfg.sys_name, cfg)
    # 5. writes to naf
    nafh.add_ext_references(news_items_with_refs, cfg.this_naf_indir(), cfg.this_naf_outdir())
