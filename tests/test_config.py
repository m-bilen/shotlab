from shotlab.config import DATA_DIR, ROOT

def test_root_is_project_folder():
    assert (ROOT / "shotlab" / "config.py").exists()
    assert DATA_DIR == ROOT / "data"