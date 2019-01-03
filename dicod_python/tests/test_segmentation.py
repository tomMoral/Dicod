import pytest
import numpy as np

from dicod_python.utils.segmentation import Segmentation


def test_segmentation_coverage():
    sig_shape = (108, 53)

    for h_seg in [5, 7, 9, 13, 17]:
        for w_seg in [3, 11]:
            z = np.zeros(sig_shape)
            segments = Segmentation((h_seg, w_seg), signal_shape=sig_shape)
            seg_slice = segments.get_seg_slice(0)
            z[seg_slice] += 1
            i_seg = segments.increment_seg(0)
            while i_seg != 0:
                seg_slice = segments.get_seg_slice(i_seg)
                z[seg_slice] += 1
                i_seg = segments.increment_seg(i_seg)

            assert np.all(z == 1)


def test_touched_segments():
    """Test detection of touched segments and records of active segments
    """
    rng = np.random.RandomState(42)

    H, W = sig_shape = (108, 53)
    n_seg = (9, 3)
    for h_radius in [5, 7, 9]:
        for w_radius in [3, 11]:
            for _ in range(20):
                h0 = rng.randint(-h_radius, sig_shape[0] + h_radius)
                w0 = rng.randint(-w_radius, sig_shape[1] + w_radius)
                z = np.zeros(sig_shape)
                segments = Segmentation(n_seg, signal_shape=sig_shape)

                touched_slice = (
                    slice(max(0, h0 - h_radius), min(H, h0 + h_radius + 1)),
                    slice(max(0, w0 - w_radius), min(W, w0 + w_radius + 1))
                )
                z[touched_slice] = 1

                touched_segments = segments.get_touched_segments(
                    (h0, w0), (h_radius, w_radius))
                segments.set_inactive_segments(touched_segments)
                n_active_segments = segments._n_active_segments

                expected_n_active_segments = segments.effective_n_seg
                for i_seg in range(segments.effective_n_seg):
                    seg_slice = segments.get_seg_slice(i_seg)
                    is_touched = np.any(z[seg_slice] == 1)
                    expected_n_active_segments -= is_touched

                    assert segments.is_active_segment(i_seg) != is_touched
                assert n_active_segments == expected_n_active_segments

    # Check an error is returned when touched radius is larger than seg_size
    segments = Segmentation(n_seg, signal_shape=sig_shape)
    with pytest.raises(ValueError, message="too large"):
        segments.get_touched_segments((0, 0), (30, 2))
