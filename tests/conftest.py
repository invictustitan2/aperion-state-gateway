"""Test configuration for pytest."""

import pytest


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "conformance: API contract conformance tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
