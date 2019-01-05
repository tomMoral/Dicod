import numpy as np


class Segmentation:
    """Segmentation of a multi-dimensional signal and utilities to navigate it.

    Parameters
    ----------
    n_seg : int or list of int
        Number of segments to use for each dimension. If only one int is
        given, use this same number for all axis.
    signal_shape : list of int or None
        Size of the considered signal.
    outer_bounds : list of (int, int)
        Outer boundaries of the full signal in case of nested segmentation.
    """

    def __init__(self, n_seg, signal_shape=None, outer_bounds=None):

        # Get the shape of the signal from signal_shape or outer_bounds
        if outer_bounds is not None:
            signal_shape_ = [v[0] for v in np.diff(outer_bounds, axis=1)]
            assert signal_shape is None or signal_shape == signal_shape_, (
                "Incoherent shape for outer_bounds and signal_shape. Got "
                "signal_shape={} and outer_bounds={}".format(
                    signal_shape, outer_bounds
                ))
            signal_shape = signal_shape_
        else:
            assert signal_shape is not None, (
                "either signal_shape or outer_bounds should be provided")
            if isinstance(signal_shape, int):
                signal_shape = [signal_shape]
            outer_bounds = [[0, s] for s in signal_shape]
        self.signal_shape = signal_shape
        self.outer_bounds = outer_bounds

        # compute size of each segments
        self.compute_seg_info(n_seg)

        # Initializes variable to keep track of active segments
        self._n_active_segments = self.effective_n_seg
        self._active_segments = [True] * self.effective_n_seg

    def compute_seg_info(self, n_seg):
        """Compute the number of segment and their shapes for each axis.
        """

        if isinstance(n_seg, int):
            n_seg = [n_seg] * len(self.signal_shape)

        self.effective_n_seg = 1
        seg_shape, n_seg_per_axis = [], []
        for size_axis, n_seg_axis in zip(self.signal_shape, n_seg):
            seg_shape.append(max(size_axis // n_seg_axis, 1))
            n_seg_per_axis.append(size_axis // seg_shape[-1])
            self.effective_n_seg *= int(n_seg_per_axis[-1])

        self.n_seg_per_axis = n_seg_per_axis
        self.seg_shape = seg_shape

    def get_seg_bounds(self, i_seg):
        """Return a segment's boundaries."""

        seg_bounds = []
        axis_offset = self.effective_n_seg
        for n_seg_axis, seg_size_axis, (axis_start, axis_end) in zip(
                self.n_seg_per_axis, self.seg_shape, self.outer_bounds):
            axis_offset //= n_seg_axis
            axis_i_seg = i_seg // axis_offset
            axis_bound_start = axis_start + axis_i_seg * seg_size_axis
            axis_bound = [axis_bound_start, axis_bound_start + seg_size_axis]
            if (axis_i_seg + 1) % n_seg_axis == 0:
                axis_bound[1] = axis_end
            seg_bounds.append(axis_bound)
            i_seg %= axis_offset
        return seg_bounds

    def get_seg_slice(self, i_seg):
        """Return a segment's slice"""
        seg_bounds = self.get_seg_bounds(i_seg)
        return (Ellipsis,) + tuple([slice(s, e) for s, e in seg_bounds])

    def get_seg_shape(self, i_seg):
        """Return a segment's shape"""
        seg_bounds = self.get_seg_bounds(i_seg)
        return tuple([e - s for s, e in seg_bounds])

    def find_segment(self, pt):
        """Find the indice of the segment containing the given point.

        If the point is not contained in the segmentation boundaries, return
        the indice of the closest segment in manhattan distance.

        Parameter
        ---------
        pt : list of int
            Coordinate of the given update.

        Return
        ------
        i_seg : int
            Indices of the segment containing pt or the closest one in
            manhattan distance if pt is out of range.
        """
        i_seg = 0
        axis_offset = self.effective_n_seg
        for x, n_seg_axis, seg_size_axis, (axis_start, axis_end) in zip(
                pt, self.n_seg_per_axis, self.seg_shape, self.outer_bounds):
            axis_offset //= n_seg_axis
            axis_i_seg = max(min((x - axis_start) // seg_size_axis,
                                 n_seg_axis - 1), 0)
            i_seg += axis_i_seg * axis_offset

        return i_seg

    def increment_seg(self, i_seg):
        """Return the next segment indice in a cyclic way."""
        return (i_seg + 1) % self.effective_n_seg

    def get_touched_segments(self, pt, radius):
        """Return all segments touched by an update in pt with a given radius.

        Parameter
        ---------
        pt : list of int
            Coordinate of the given update.
        radius: int or list of int
            Radius of the update. If an integer is given, use the same integer
            for all axis.

        Return
        ------
        segments : list of int
            Indices of all segments touched by this update, including the one
            in which the update took place.
        """
        if isinstance(radius, int):
            radius = [radius] * len(pt)

        for r, size_axis in zip(radius, self.seg_shape):
            if r >= size_axis:
                raise ValueError("Interference radius is too large compared "
                                 "to the segmentation size.")

        i_seg = self.find_segment(pt)
        seg_bounds = self.get_seg_bounds(i_seg)

        segments = [i_seg]
        axis_offset = self.effective_n_seg
        for x, r, n_seg_axis, (axis_start, axis_end) in zip(
                pt, radius, self.n_seg_per_axis, seg_bounds):
            axis_offset //= n_seg_axis
            axis_i_seg = i_seg // axis_offset
            i_seg %= axis_offset
            new_segments = []
            if x - r < axis_start and axis_i_seg > 0:
                new_segments.extend([n - axis_offset for n in segments])
            if x + r >= axis_start or x - r < axis_end:
                new_segments.extend([n for n in segments])
            if x + r >= axis_end and axis_i_seg < n_seg_axis - 1:
                new_segments.extend([n + axis_offset for n in segments])
            segments = new_segments

        for ii_seg in segments:
            msg = ("Segment indice out of bound. Got {} for effective n_seg {}"
                   .format(ii_seg, self.effective_n_seg))
            assert ii_seg < self.effective_n_seg, msg

        return segments

    def is_active_segment(self, i_seg):
        """Return True if segment i_seg is active"""
        return self._active_segments[i_seg]

    def set_active_segments(self, indices):
        """Activate segments indices and return the number of changed status.
        """
        if isinstance(indices, int):
            indices = [indices]

        n_changed_status = 0
        for i_seg in indices:
            n_changed_status += not self._active_segments[i_seg]
            self._active_segments[i_seg] = True

        self._n_active_segments += n_changed_status
        assert self._n_active_segments <= self.effective_n_seg

        return n_changed_status

    def set_inactive_segments(self, indices):
        """Deactivate segments indices and return the number of changed status.
        """
        if isinstance(indices, int):
            indices = [indices]

        n_changed_status = 0
        for i_seg in indices:
            n_changed_status += self._active_segments[i_seg]
            self._active_segments[i_seg] = False

        self._n_active_segments -= n_changed_status
        return self._n_active_segments >= 0

        return n_changed_status

    def exist_active_segment(self):
        """Return True if at least one segment is active."""
        return self._n_active_segments > 0

    def test_active_segment(self, dz, tol):
        """Test the state of active segments is coherent with dz and tol
        """
        for i in range(self.effective_n_seg):
            if not self.is_active_segment(i):
                seg_slice = self.get_seg_slice(i)
                assert np.all(abs(dz[seg_slice]) <= tol)
