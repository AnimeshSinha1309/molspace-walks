import pytest


def test_testing_setup():
    assert True


@pytest.mark.xfail
def test_tests_failing():
    assert False
