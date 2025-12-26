"""Pytest configuration for CLEANSER tests."""

import sys
import io
import warnings


# Suppress the I/O error during pytest cleanup by wrapping the problematic operation
original_snap = None


def _patched_snap(self):
    """Patched snap method that handles closed files gracefully."""
    try:
        return original_snap(self)
    except ValueError as e:
        if "I/O operation on closed file" in str(e):
            # Return empty string if file is closed
            return ""
        raise


def pytest_configure(config):
    """Configure pytest hooks."""
    global original_snap
    try:
        # Patch the capture.snap method to handle closed files
        from _pytest import capture as pytest_capture
        if hasattr(pytest_capture, 'FDCapture'):
            original_snap = pytest_capture.FDCapture.snap
            pytest_capture.FDCapture.snap = _patched_snap
    except (ImportError, AttributeError):
        pass
