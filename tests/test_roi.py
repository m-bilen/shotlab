from shotlab.roi import get_roi, save_roi

def test_save_and_get_roi(tmp_path):
    path = tmp_path / "rois.csv"
    save_roi("a.mp4", (10, 20, 300, 300), path)
    save_roi("b.mp4", (1, 2, 50, 50), path)
    save_roi("a.mp4", (11, 21, 310, 310), path)

    assert get_roi("a.mp4", path) == (11, 21, 310, 310)
    assert get_roi("b.mp4", path) == (1, 2, 50, 50)

def test_load_rois_handles_empty_file(tmp_path):
    path = tmp_path / "rois.csv"
    path.write_text("")

    save_roi("a.mp4", (10, 20, 300, 300), path)
    assert get_roi("a.mp4", path) == (10, 20, 300, 300)