from shotlab.geometry import angle, square_roi

def test_square_roi_keeps_square_box():
    assert square_roi((100, 100, 200, 200), 1920,1080) == (100, 100, 200, 200)

def test_square_roi_expands_tall_box_around_center():
    assert square_roi((100,100,200,400), 1920,1080) == (0,100,400,400)

def test_square_roi_stays_inside_frame():
    assert square_roi((1800,900,100,300), 1920,1080) == (1620,780,300,300)

def test_angle_right_and_straight():
    assert round(float(angle((1, 0), (0, 0), (0, 1))), 6) == 90.0
    assert round(float(angle((-1, 0), (0, 0), (1, 0))), 6) == 180.0

def test_angle_many_points():
    result = angle([(1, 0), (-1, 0)], [(0, 0), (0, 0)], [(0, 1), (1, 0)])
    assert result.round(6).tolist() == [90.0, 180.0]
