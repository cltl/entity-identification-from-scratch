import pytest
from config import Config


def test_config_creation():
    yml_file = 'cfg/dbpedia_abstracts.yml'
    cfg = Config(yml_file)
    assert cfg.corpus_name == 'dbpedia_abstracts'
    assert cfg.max_documents is None
    assert len(cfg.factors) == 2
    assert cfg.sys_dir == 'data/dbpedia_abstracts/system'
