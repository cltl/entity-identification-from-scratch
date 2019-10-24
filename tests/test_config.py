from config import Config
import pytest
from path import Path


def test_config_creation_fails_if_input_file_does_not_exist():
    cfg = Config('tests/cfg/test.yml')
    assert cfg.experiment_dir == 'tests/data/nif35_2'

    file_not_existing = "some_wrong_path/abstracts_nl35.ttl"
    assert cfg.raw_input == file_not_existing
    with pytest.raises(FileNotFoundError):
        cfg.setup_input()


def test_project_structure():
    cfg = Config('tests/cfg/test.yml')
    cfg.create_sysdirs()
    assert Path("{}/{}".format(cfg.experiment_dir, cfg.models_dir)).exists()
    assert Path(cfg.this_naf_indir()).exists()
    assert Path(cfg.this_naf_outdir()).exists()

    # create_sysdirs allows for already existing directories
    cfg.create_sysdirs()
    assert Path("{}/{}".format(cfg.experiment_dir, cfg.models_dir)).exists()
    assert Path(cfg.this_naf_indir()).exists()
    assert Path(cfg.this_naf_outdir()).exists()


