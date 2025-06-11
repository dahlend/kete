"""
Broken Power Law

This may be used to draw samples fairly from a broken power law.
"""

from __future__ import annotations

from functools import partial

import numpy as np
from numpy.typing import NDArray


class CumulativePowerLaw:
    """
    Representation of a cumulative power law made up of individual power law segments.

    The total function is made up of a set of power law segments with their own slopes
    which are made to meet correctly at the specified cutoff values.

    Parameters
    ----------
    slopes:
        The slopes of the different sections of the power law.
    cutoffs:
        The break points between the different sections of the power laws.
    num_larger_than_1:
        The number of objects larger than 1, this is the single offset required to
        calculate the offsets of each of the individual power law segments.
    max_possible:
        The maximum value possible, any value larger than this will return a total of 0,
        regardless of slopes.
    delta:
        Curvature parameter which adjusts how rounded the joints are between the
        individual power law sections.
    """

    def __init__(self, slopes, cutoffs, num_larger_than_1, max_possible, delta=0.2):
        self.slopes = slopes
        self.cutoffs = np.array(cutoffs, dtype=float)
        self.num_larger_than_1 = num_larger_than_1
        self.max_possible = max_possible
        self.delta = delta

        # Calculate the halfway point between the cutoffs in log space.
        self.cutoff_mids = 10 ** (
            (np.log10(self.cutoffs[1:]) + np.log10(self.cutoffs[:-1])) / 2.0
        )
        self.slope_pairs = list(zip(slopes, slopes[1:]))

        # Given the slopes, cutoffs, and number of values larger than 1, calculate the
        # offset values for each slope region which will make the regions meet correctly
        # at the boundaries.
        one_region = np.digitize(1, self.cutoff_mids)

        offsets = [
            num_larger_than_1
            / _smooth_power_law_segment(
                1.0, *self.slope_pairs[one_region], self.cutoffs[one_region], 1.0, delta
            )
        ]

        # Calculate offsets for all power law regions greater than 1
        for region in range(one_region, len(self.cutoff_mids)):
            cur_region = _smooth_power_law_segment(
                self.cutoff_mids[region],
                *self.slope_pairs[region],
                self.cutoffs[region],
                offsets[-1],
                delta,
            )
            next_region = _smooth_power_law_segment(
                self.cutoff_mids[region],
                *self.slope_pairs[region + 1],
                self.cutoffs[region + 1],
                1.0,
                delta,
            )
            offsets.append(cur_region / next_region)
        # Calculate offsets for all power law regions less than 1
        for region in range(one_region):
            region = one_region - region - 1
            cur_region = _smooth_power_law_segment(
                self.cutoff_mids[region],
                *self.slope_pairs[region + 1],
                self.cutoffs[region + 1],
                offsets[0],
                delta,
            )
            next_region = _smooth_power_law_segment(
                self.cutoff_mids[region],
                *self.slope_pairs[region],
                self.cutoffs[region],
                1.0,
                delta,
            )
            offsets.insert(0, cur_region / next_region)
        self.offsets = np.array(offsets)

        self.partials = []
        self.partials_der = []
        for i, (a, b) in enumerate(self.slope_pairs):
            self.partials.append(
                partial(
                    _smooth_power_law_segment,
                    a=a,
                    b=b,
                    x_b=self.cutoffs[i],
                    offset=self.offsets[i],
                    delta=delta,
                )
            )

            self.partials_der.append(
                partial(
                    _smooth_power_law_segment_der,
                    a=a,
                    b=b,
                    x_b=self.cutoffs[i],
                    offset=self.offsets[i],
                    delta=delta,
                )
            )

    def num_larger_than(self, value: NDArray[np.floating]):
        """
        Calculate the cumulative values of objects larger than the specified value.

        Parameters
        ----------
        value:
            A list of values.
        """
        value = np.array(value)

        # Calculate the value for each element
        idx = np.digitize(value, self.cutoff_mids)
        vals = np.array([self.partials[i](x) for i, x in zip(idx, value)])
        vals[value > self.max_possible] = 0
        return vals

    def value_of_nth_largest(self, nth_largest: list) -> list:
        """
        Calculate the value of the n-th largest.

        Parameters
        ----------
        nth_largest:
            A list of indices of the nth largest, this does not have to be an int.
        """
        nth_largest = np.array_split(
            np.array(nth_largest), int(np.ceil(len(nth_largest) / 1000))
        )
        cutoffs = np.array(list(self.cutoff_mids) + [np.inf])

        all_values = []
        for batch in nth_largest:
            batch_vals = []
            for p, p_d in zip(self.partials, self.partials_der):
                value = np.ones_like(batch) * 0.01
                for _ in range(10000):
                    value[value < 0] = 0.001
                    step = (p(value) - batch) / p_d(value)
                    if max(abs(step)) < 1e-10:
                        break
                    value = value - 0.1 * step
                batch_vals.append(value)
            all_values.append(batch_vals)
        all_values_arr = np.hstack(all_values)

        return [
            all_values_arr[d, idx]
            for idx, d in enumerate(
                np.argmax(all_values_arr < cutoffs[:, np.newaxis], axis=0)
            )
        ]

    def sample(self, min_value, max_value=np.inf, scale=1.0, n_samples=None, seed=42):
        """
        Sample the distribution.

        This returns a sorted array of values which are constructed using Inverse
        Transform Sampling from the fit.

        Parameters
        ----------
        min_value : float
            The minimum value.
        max_value : float
            The maximum value.
        scale : float
            The relative fraction of the total population to sample from, for example,
            if this value is set to 0.5, then the total count of each value range
            will be half of the expected true population count. This makes it so
            sub-populations may be sampled assuming they all are relative fractions of
            the same total distribution.
        n_samples : int
            If this is set to a value, then scale must be set to None.
            This will sample exactly this number of samples fairly, if this is set to
            None, this will sample the expected number of values between the min and
            maximum specified, scaled by the `scale` key word.
        seed : int
            The random seed used for all generation of random numbers. This can be used
            to accurately reconstruct the same objects.
        """
        rng = np.random.default_rng(seed)
        if n_samples is not None and scale is not None:
            raise ValueError("Either scale may be set or n_objects, but not both.")

        max_index, min_index = self.num_larger_than([min_value, max_value])

        if n_samples is None:
            n_samples = int((max_index - min_index) * scale)

        rand_idx = rng.random(n_samples) * (max_index - min_index) + min_index
        values = self.value_of_nth_largest(rand_idx)
        return np.array(sorted(values, reverse=True))

    def __repr__(self):
        return (
            type(self).__name__
            + f"(slopes={list(self.slopes)},\n\tcutoffs={list(self.cutoffs)},"
            f"\n\num_larger_than_1={self.num_larger_than_1},"
            f"\n\tmax_possible={self.max_possible})"
        )


def _smooth_power_law_segment(x, a, b, x_b, offset, delta):
    """
    A broken power law curve evaluated at the position x.

    Parameters
    ----------
    x:
        Point on the power law curve to evaluate.
    a:
        Slope of the power law before the break point.
    b:
        Slope of the power law after the break point.
    x_b:
        The x position of the break point.
    offset:
        The value of the curve at the `x_b` point.
    delta:
        Rounding term which changes how curved the break point is.
    """
    return (
        offset
        * (x / x_b) ** (-a)
        * (0.5 + 0.5 * (x / x_b) ** (1 / delta)) ** ((a - b) * delta)
    )


def _smooth_power_law_segment_der(x, a, b, x_b, offset, delta):
    """
    The derivative of the power law segment defined above.

    Parameters
    ----------
    x:
        Point on the power law curve to evaluate.
    a:
        Slope of the power law before the break point.
    b:
        Slope of the power law after the break point.
    x_b:
        The x position of the break point.
    offset:
        The value of the curve at the `x_b` point.
    delta:
        Rounding term which changes how curved the break point is.
    """
    x = x / x_b
    return offset * (
        x ** (-a - 1)
        * (-(2 ** (delta * (b - a))))
        * (x ** (1 / delta) + 1) ** (a * delta - b * delta - 1)
        * (a + b * x ** (1 / delta))
    )
