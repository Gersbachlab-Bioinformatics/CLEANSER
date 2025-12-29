"""Pytest configuration for CLEANSER tests."""

import sys
import io
import warnings
import pytest
import contextlib


def pytest_configure(config):
    """Configure pytest hooks to handle Python 3.14 I/O issues."""
    try:
        # Patch contextlib._GeneratorContextManager to handle closed file errors
        original_enter = contextlib._GeneratorContextManager.__enter__
        
        def patched_enter(self):
            """Patched __enter__ that handles closed file errors gracefully."""
            try:
                return original_enter(self)
            except ValueError as e:
                if "I/O operation on closed file" in str(e):
                    # Return None or handle gracefully
                    return None
                raise
        
        contextlib._GeneratorContextManager.__enter__ = patched_enter
        
        # Also patch FDCapture.snap to handle closed files during pytest cleanup
        from _pytest import capture as pytest_capture
        original_snap = pytest_capture.FDCapture.snap
        
        def patched_snap(self):
            """Patched snap method that handles closed files gracefully."""
            try:
                return original_snap(self)
            except ValueError as e:
                if "I/O operation on closed file" in str(e):
                    return ""
                raise
        
        pytest_capture.FDCapture.snap = patched_snap
    except Exception:
        pass

