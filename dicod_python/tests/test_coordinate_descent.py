import pytest
import numpy as np

from dicod_python.coordinate_descent_2d import _get_seg_info
from dicod_python.coordinate_descent_2d import _is_interfering_update


def test_is_interfering_update():
    n_times = 100
    n_times_atom = 4
    n_times_update = 2 * n_times_atom - 1

    t_start, t_end = 0, n_times
    # Test without segment
    update = np.ones(n_times_update)
    update[n_times_atom - 1] = 2
    for t in range(n_times):
        signal = np.zeros(n_times)
        t_start_up = max(0, t - n_times_atom + 1)
        t_end_up = min(t + n_times_atom, n_times)

        ll = t_end_up - t_start_up
        offset = max(0, n_times_atom - t - 1)
        signal[t_start_up:t_end_up] = update[offset:offset + ll]

        assert signal[t] == 2

        _, down, up = _is_interfering_update(t, n_times_atom, t_start, t_end)
        assert up == (offset > 0), "bad interference detection upstream"
        down_truth = (ll < n_times_update and offset == 0)
        assert down == down_truth, "bad interference detection downstream"


def test_get_seg_info():
    height_valid, width_valid = 53, 201
    atom_shape = 10, 37

    seg_shape, grid_seg, effective_n_seg = _get_seg_info(
        'auto', height_valid, width_valid, atom_shape)
    height_n_seg, width_n_seg = grid_seg
    height_seg, width_seg = seg_shape

    assert effective_n_seg == height_n_seg * width_n_seg

    # Assert that no segment is empty and that all coordinates are covered
    assert (height_n_seg - 1) * height_seg < height_valid - 1
    assert height_valid - 1 < height_n_seg * height_seg
    assert (width_n_seg - 1) * width_seg < width_valid - 1
    assert width_valid - 1 < width_n_seg * width_seg

    for n_seg in range(1, 100):
        seg_shape, grid_seg, effective_n_seg = _get_seg_info(
            n_seg, height_valid, width_valid, atom_shape)
        height_n_seg, width_n_seg = grid_seg
        height_seg, width_seg = seg_shape

        assert effective_n_seg == height_n_seg * width_n_seg

        # Assert that no segment is empty and that all coordinates are covered
        assert (height_n_seg - 1) * height_seg <= height_valid - 1
        assert height_valid - 1 < height_n_seg * height_seg
        assert (width_n_seg - 1) * width_seg <= width_valid - 1
        assert width_valid - 1 < width_n_seg * width_seg
