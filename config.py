import yaml
from path import Path
import os.path


class Config:
    def __init__(self, yml):
        with open(yml, 'r') as f:
            cfg = yaml.full_load(f)
        self.max_documents = cfg['max_documents']
        self.corpus_name = cfg['corpus_name']
        self.factors = cfg['factors']
        self.bert_model = cfg['bert_model']
        self.sys_name = cfg['sys_name']
        self.modify_entities = cfg['modify_entities']
        self.ner = cfg['ner']
        self.uri_prefix = cfg['uri_prefix']
        self.naf_entity_layer = cfg['naf_entity_layer']
        self.corpus_dir = self.corpus_name
        if self.max_documents:
            self.corpus_dir += '_{}'.format(self.max_documents)
        self.corpus_uri = 'http://{}.nl'.format(self.corpus_dir)
        self.data_dir = 'data/{}'.format(self.corpus_dir)
        self.raw_input_dir = 'data/{}/input_data'.format(self.corpus_name)
        self.raw_input = '{}/{}'.format(self.raw_input_dir, cfg['raw_input'])
        self.input_dir = '{}/documents'.format(self.data_dir)
        self.sys_dir = '{}/system'.format(self.data_dir)
        self.this_sys_dir = '{}/{}'.format(self.sys_dir, self.sys_name)
        self.emb_dir = '{}/emb'.format(self.data_dir)
        self.naf_dir = '{}/naf'.format(self.this_sys_dir)
        self.el_file = Path('{}/el.pkl'.format(self.this_sys_dir))
        self.graphs_file = Path('{}/graphs.graph'.format(self.this_sys_dir))

    def setup_input(self):
        if not os.path.exists(Path(self.raw_input)):
            raise OSError('missing expected input file: {}'.format(self.raw_input))
        input_dir_path = Path(self.input_dir)
        if not os.path.exists(input_dir_path):
            input_dir_path.makedirs()


def create(yml):
    cfg = Config(yml)
    input_dir_path = Path(cfg.input_dir)

    if not os.path.exists(input_dir_path):
        raise OSError('expected input directory not found: {}'.format(input_dir_path))

    naf_dir_path = Path(cfg.naf_dir)
    if not os.path.exists(naf_dir_path):
        naf_dir_path.makedirs()

    emb_dir_path = Path(cfg.emb_dir)
    if not os.path.exists(emb_dir_path):
        emb_dir_path.mkdir()

    return cfg
