import pathlib
import yaml
from path import Path


class Config:
    def __init__(self, yml):
        with open(yml, 'r') as f:
            cfg = yaml.full_load(f)
        self.max_documents = cfg['max_documents']
        self.min_text_length = cfg['min_text_length']
        # for make_x_corpus.py
        self.raw_input = cfg['raw_input']

        self.experiment_dir = cfg['experiment_dir']
        self.factors = cfg['factors']
        self.bert_model = cfg['bert_model']
        self.sys_name = cfg['sys_name']
        self.modify_entities = cfg['modify_entities']
        self.ner = cfg['ner']
        self.uri_prefix = cfg['uri_prefix']
        self.models_dir = cfg['models_dir']
        self.naf_indir = cfg['naf_indir']
        self.naf_outdir = cfg['naf_outdir']
        # created by make_x_corpus.py, loaded by main.py/baselines.py
        self.news_items_file_name = cfg['news_items_file']
        self.el_file = cfg['el_file']
        self.graphs_file = cfg['graphs_file']
        self.doc2vec_ids = cfg['doc2vec_ids']
        self.doc2vec_model = cfg['doc2vec_model']
        self.corpus_name = cfg['corpus_name']
        self.create_input_nafs = cfg['create_input_nafs']

    def news_items_file(self):
        return "{}/{}".format(self.experiment_dir, self.news_items_file_name)

    def this_sys_dir(self):
        return '{}/{}'.format(self.experiment_dir, self.sys_name)

    def this_naf_indir(self):
        return '{}/{}/{}'.format(self.experiment_dir, self.sys_name, self.naf_indir)

    def this_naf_outdir(self):
        return '{}/{}/{}'.format(self.experiment_dir, self.sys_name, self.naf_outdir)

    def el_file_path(self):
        return '{}/el.pkl'.format(self.this_sys_dir())

    def graphs_file_path(self):
        return '{}/graphs.graph'.format(self.this_sys_dir())

    def doc2vec_ids_path(self):
        return '{}/{}/{}'.format(self.experiment_dir, self.models_dir, self.doc2vec_ids)

    def doc2vec_model_path(self):
        return '{}/{}/{}'.format(self.experiment_dir, self.models_dir, self.doc2vec_model)

    def create_sysdirs(self):
        """Creates input and output naf dirs and embedding models dir.

        Allows pre-existence"""
        create_dir(self.this_naf_indir())
        create_dir(self.this_naf_outdir())
        create_dir('{}/{}'.format(self.experiment_dir, self.models_dir))

    def setup_input(self):
        """loads a raw input file into a news-items pickle file in the experiment directory

        used by make_x_corpus.py

        Requires: a raw_input file
        Assumes: the experiment directory does not exist yet"""
        if not Path(self.raw_input).exists():
            raise FileNotFoundError("{} does not exist".format(self.raw_input))
        if Path(self.experiment_dir).exists():
            raise ValueError("The experiment directory already exists. Delete it to run this setup.")
        pathlib.Path(self.experiment_dir).mkdir(exist_ok=False)


def create_dir(dir_name):
    pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)


def load(cfg_file):
    """creates config for main.py/baselines.py"""
    cfg = Config(cfg_file)
    return cfg
