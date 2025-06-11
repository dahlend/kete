import os

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--horizons",
        action="store_true",
        default=False,
        help="Run the optional horizons tests, this requires an internet connection.",
    )


def pytest_runtest_setup(item):
    if "horizons" in item.keywords and not item.config.getoption("horizons"):
        pytest.skip("need --horizons option to run this test")


@pytest.fixture
def data_path(request):
    return os.path.join(os.path.dirname(request.path), "data/")
