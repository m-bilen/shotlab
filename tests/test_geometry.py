from shotlab.geometry import square_roi

def test_square_roi_keeps_square_box():
    assert square_roi((100, 100, 200, 200), 1920,1080) == (100, 100, 200, 200)

def test_square_roi_expands_tall_box_around_center():
    assert square_roi((100,100,200,400), 1920,1080) == (0,100,400,400)

def test_square_roi_stays_inside_frame():
    assert square_roi((1800,900,100,300), 1920,1080) == (1620,780,300,300)