"""Shared fixtures for the comb-content analysis tests."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from combinator.content import data  # noqa: E402


@pytest.fixture(scope="session")
def tabs():
    return data.load_tables()


@pytest.fixture(scope="session")
def masks():
    return data.load_masks(keys=("Contents", "ContentsCrop"))


@pytest.fixture(scope="session")
def contents(masks):
    return masks["Contents"]


@pytest.fixture(scope="session")
def px_to_cm(masks):
    return data.px_to_cm(masks["ContentsCrop"])


@pytest.fixture(autouse=True)
def seeded():
    """Fix the global numpy stream so bootstrap smoke tests are stable."""
    np.random.seed(1234)
    yield


def assert_stats(observed: dict, mean: float, sd: float,
                 tol: float = 0.002, sd_tol: float = 0.003):
    """Assert a paper-style ``mean ± sd`` stat against a computed dict.

    ``observed`` carries ``mean``/``sd`` keys; the tolerances absorb the
    paper's 3-decimal rounding (e.g. SD 0.1445 reported as 0.145).
    """
    assert abs(observed["mean"] - mean) <= tol
    assert abs(observed["sd"] - sd) <= sd_tol
