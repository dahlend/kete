# SPDX-FileCopyrightText: 2026 Dar Dahlen
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from kete import Data


def test_quantile():
    data = Data([3.0, 1.0, 2.0, 4.0, 5.0])
    assert data.quantile(0.5) == 3.0
    assert data.quantile(0.125) == 1.5
    # Quantiles outside of 0 to 1 are clamped.
    assert data.quantile(-1.0) == 1.0
    assert data.quantile(2.0) == 5.0


def test_quantile_nan_raises():
    with pytest.raises(ValueError, match="NaN"):
        Data([1.0, 2.0, 3.0]).quantile(float("nan"))
