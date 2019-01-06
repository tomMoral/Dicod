import pytest
import numpy as np

from dicod_python.utils.segmentation import Segmentation


def test_segmentation_coverage():
    sig_shape = (108, 53)

    for h_seg in [5, 7, 9, 13, 17]:
        for w_seg in [3, 11]:
            z = np.zeros(sig_shape)
            segments = Segmentation(n_seg=(h_seg, w_seg),
                                    signal_shape=sig_shape)
            assert tuple(segments.n_seg_per_axis) == (h_seg, w_seg)
            seg_slice = segments.get_seg_slice(0)
            seg_shape = segments.get_seg_shape(0)
            assert seg_shape == z[seg_slice].shape
            z[seg_slice] += 1
            i_seg = segments.increment_seg(0)
            while i_seg != 0:
                seg_slice = segments.get_seg_slice(i_seg)
                seg_shape = segments.get_seg_shape(i_seg)
                assert seg_shape == z[seg_slice].shape
                z[seg_slice] += 1
                i_seg = segments.increment_seg(i_seg)

            assert np.all(z == 1)


def test_segmentation_coverage_overlap():
    sig_shape = (505, 407)

    for overlap in [(3, 0), (0, 5), (3, 5), (12, 7)]:
        for h_seg in [5, 7, 9, 13, 15, 17]:
            for w_seg in [3, 11]:
                segments = Segmentation(n_seg=(h_seg, w_seg),
                                        signal_shape=sig_shape,
                                        overlap=overlap)
                z = np.zeros(sig_shape)
                for i_seg in range(segments.effective_n_seg):
                    seg_slice = segments.get_seg_slice(i_seg, only_inner=True)
                    z[seg_slice] += 1
                    i_seg = segments.increment_seg(i_seg)
                non_overlapping = np.prod(sig_shape)
                assert np.sum(z == 1) == non_overlapping

                z = np.zeros(sig_shape)
                for i_seg in range(segments.effective_n_seg):
                    seg_slice = segments.get_seg_slice(i_seg)
                    z[seg_slice] += 1
                    i_seg = segments.increment_seg(i_seg)

                h_ov, w_ov = overlap
                h_seg, w_seg = segments.n_seg_per_axis
                expected_overlap = ((h_seg - 1) * sig_shape[1] * 2 * h_ov)
                expected_overlap += ((w_seg - 1) * sig_shape[0] * 2 * w_ov)

                # Compute the number of pixel where there is more than 2
                # segments overlappping.
                corner_overlap = 4 * (h_seg - 1) * (w_seg - 1) * h_ov * w_ov
                expected_overlap -= 2 * corner_overlap

                non_overlapping -= expected_overlap + corner_overlap
                assert non_overlapping == np.sum(z == 1)
                assert expected_overlap == np.sum(z == 2)
                assert corner_overlap == np.sum(z == 4)


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


def test_change_coordinate():
    sig_shape = (505, 407)
    overlap = (12, 7)
    n_seg = (4, 4)
    segments = Segmentation(n_seg=n_seg, signal_shape=sig_shape,
                            overlap=overlap)

    for i_seg in range(segments.effective_n_seg):
        seg_bound = segments.get_seg_bounds(i_seg)
        seg_shape = segments.get_seg_shape(i_seg)
        origin = tuple([start for start, _ in seg_bound])
        assert segments.get_global_coordinate(i_seg, (0, 0)) == origin
        assert segments.get_local_coordinate(i_seg, origin) == (0, 0)

        corner = tuple([end for _, end in seg_bound])
        assert segments.get_global_coordinate(i_seg, seg_shape) == corner
        assert segments.get_local_coordinate(i_seg, corner) == seg_shape
