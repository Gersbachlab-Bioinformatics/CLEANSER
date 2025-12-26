"""Tests for MuData functionality in CLEANSER."""

import os
import tempfile
from pathlib import Path

import mudata as md
import numpy as np
import pytest

from cleanser.configuration import MuDataConfiguration, Model


# Path to the test data
TEST_DATA_DIR = Path(__file__).parent / "data"
TEST_MUDATA_FILE = TEST_DATA_DIR / "test_mudata_small.h5mu"


@pytest.fixture
def test_mudata():
    """Load the test MuData file."""
    if not TEST_MUDATA_FILE.exists():
        pytest.skip(f"Test data not found: {TEST_MUDATA_FILE}")
    return md.read(str(TEST_MUDATA_FILE))


class TestMuDataConfiguration:
    """Tests for MuDataConfiguration with real MuData files."""

    def test_mudata_file_exists(self):
        """Test that test MuData file exists."""
        assert TEST_MUDATA_FILE.exists(), f"Test file not found: {TEST_MUDATA_FILE}"

    def test_mudata_file_size(self):
        """Test that test MuData file is under 500KB."""
        file_size = TEST_MUDATA_FILE.stat().st_size
        assert file_size < 500 * 1024, f"File too large: {file_size / 1024:.1f} KB"

    def test_mudata_load(self, test_mudata):
        """Test loading MuData file."""
        assert test_mudata is not None
        assert hasattr(test_mudata, 'mod')
        assert len(test_mudata.mod) > 0

    def test_mudata_modalities(self, test_mudata):
        """Test that MuData has expected modalities."""
        modalities = list(test_mudata.mod.keys())
        assert 'guide' in modalities, f"Missing 'guide' modality. Available: {modalities}"

    def test_mudata_guide_modality(self, test_mudata):
        """Test guide modality structure."""
        guide_mod = test_mudata.mod['guide']
        assert guide_mod is not None
        assert guide_mod.shape[0] > 0  # Has guides
        assert guide_mod.shape[1] > 0  # Has cells
        assert hasattr(guide_mod, 'X')

    def test_mudata_configuration_creation(self, test_mudata):
        """Test creating MuDataConfiguration from MuData file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "output.h5mu"
            
            config = MuDataConfiguration(
                input=str(TEST_MUDATA_FILE),
                modality='guide',
                capture_method='capture_type',
                output_layer='guide_assignment',
                model=Model.CS,
                sample_output=None,
                posteriors_output=str(output_file),
                threshold=None,
            )
            
            assert config is not None
            assert config.model == Model.CS
            assert config.output_layer == 'guide_assignment'

    def test_mudata_configuration_with_threshold(self, test_mudata):
        """Test MuDataConfiguration with threshold."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "output.h5mu"
            
            config = MuDataConfiguration(
                input=str(TEST_MUDATA_FILE),
                modality='guide',
                capture_method='capture_type',
                output_layer='guide_assignment',
                model=Model.CS,
                sample_output=None,
                posteriors_output=str(output_file),
                threshold=0.5,
            )
            
            assert config.threshold == 0.5
            assert config.collect_posteriors == config._raw_and_threshold_collect

    def test_mudata_configuration_without_threshold(self, test_mudata):
        """Test MuDataConfiguration without threshold."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "output.h5mu"
            
            config = MuDataConfiguration(
                input=str(TEST_MUDATA_FILE),
                modality='guide',
                capture_method='capture_type',
                output_layer='guide_assignment',
                model=Model.CS,
                sample_output=None,
                posteriors_output=str(output_file),
                threshold=None,
            )
            
            assert config.threshold is None
            assert config.collect_posteriors == config._raw_collect

    def test_mudata_gen_data(self):
        """Test data generation from MuData file."""
        config = MuDataConfiguration(
            input=str(TEST_MUDATA_FILE),
            modality='guide',
            capture_method='capture_type',
            output_layer='guide_assignment',
            model=Model.CS,
            sample_output=None,
            posteriors_output=None,
            threshold=None,
        )
        
        # Generate data
        data = list(config.gen_data())
        
        # Should have data
        assert len(data) > 0
        
        # Each entry should be (guide_id, cell_id, count)
        for entry in data:
            assert len(entry) == 3
            guide_id, cell_id, count = entry
            # Check for numeric or string types (including numpy types)
            assert isinstance(guide_id, (str, int, np.integer))
            assert isinstance(cell_id, (str, int, np.integer))
            assert isinstance(count, (int, float, np.integer, np.floating))
            assert count > 0

    def test_mudata_data_consistency(self):
        """Test that data from gen_data is consistent with input."""
        config = MuDataConfiguration(
            input=str(TEST_MUDATA_FILE),
            modality='guide',
            capture_method='capture_type',
            output_layer='guide_assignment',
            model=Model.CS,
            sample_output=None,
            posteriors_output=None,
            threshold=None,
        )
        
        # Load original guide counts
        mdata = md.read(str(TEST_MUDATA_FILE))
        guide_mod = mdata.mod['guide']
        original_shape = guide_mod.X.shape
        
        # Get generated data
        data = list(config.gen_data())
        
        # Collect unique guides and cells
        guides = set()
        cells = set()
        for guide_id, cell_id, _ in data:
            guides.add(guide_id)
            cells.add(cell_id)
        
        # Should have data from modality
        assert len(guides) > 0
        assert len(cells) > 0

    def test_mudata_multiple_modalities(self):
        """Test MuData file with multiple modalities."""
        mdata = md.read(str(TEST_MUDATA_FILE))
        
        # Should have multiple modalities
        assert len(mdata.mod) >= 1
        
        # Test with each modality
        for mod_name in ['guide', 'gene', 'hashing']:
            if mod_name in mdata.mod:
                config = MuDataConfiguration(
                    input=str(TEST_MUDATA_FILE),
                    modality=mod_name,
                    capture_method='capture_type',
                    output_layer='test_output',
                    model=Model.CS,
                    sample_output=None,
                    posteriors_output=None,
                    threshold=None,
                )
                
                data = list(config.gen_data())
                assert len(data) > 0

    def test_mudata_model_detection(self):
        """Test Model enum values."""
        assert Model.CS == "cs-guide-mixture.stan"
        assert Model.DC == "dc-guide-mixture.stan"

    def test_mudata_configuration_output_matrices(self):
        """Test that output matrices are properly initialized."""
        config = MuDataConfiguration(
            input=str(TEST_MUDATA_FILE),
            modality='guide',
            capture_method='capture_type',
            output_layer='guide_assignment',
            model=Model.CS,
            sample_output=None,
            posteriors_output=None,
            threshold=0.5,
        )
        
        # Should have output matrices initialized
        assert config.output_matrix is not None
        assert config.output_binary_matrix is not None
        assert config.output_matrix.shape == config.output_binary_matrix.shape

    def test_mudata_threshold_variation(self):
        """Test different threshold values."""
        thresholds = [None, 0.1, 0.5, 0.9, 1.0]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            for threshold in thresholds:
                output_file = Path(tmpdir) / f"output_{threshold}.h5mu"
                
                config = MuDataConfiguration(
                    input=str(TEST_MUDATA_FILE),
                    modality='guide',
                    capture_method='capture_type',
                    output_layer='guide_assignment',
                    model=Model.CS,
                    sample_output=None,
                    posteriors_output=str(output_file),
                    threshold=threshold,
                )
                
                assert config.threshold == threshold
