from rl_baseline.util import linear_schedule

def test_linear_scheduler():
    assert linear_schedule(0.5, 0, 100, 200, 150) == 0.25, 'In-between value of linear scheduler.'
    assert linear_schedule(0.5, 0, 100, 200, 0) == 0.5, 'Pre-interval value of linear scheduler.'
    assert linear_schedule(0.5, 0, 100, 200, 300) == 0, 'Post-interval value of linear scheduler.'
