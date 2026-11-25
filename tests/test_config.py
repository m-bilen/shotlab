from swish_or_brick.config import DATA_DIR, ROOT

def test_root_is_project_folder():
    assert (ROOT / "swish_or_brick" / "config.py").exists()
    assert DATA_DIR == ROOT / "data"