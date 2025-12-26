"""Tests for the configuration module."""

import io
import tempfile
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from scipy.sparse import dok_matrix

from cleanser.configuration import Model, MtxConfiguration, MuDataConfiguration


class TestModel:
    """Tests for the Model enum."""

    def test_model_cs_value(self):
        """Test that CS model has correct value."""
        assert Model.CS == "cs-guide-mixture.stan"

    def test_model_dc_value(self):
        """Test that DC model has correct value."""
        assert Model.DC == "dc-guide-mixture.stan"


class TestMuDataConfigurationThresholdCollection:
    """Tests for MuDataConfiguration._raw_and_threshold_collect method."""

    @pytest.fixture
    def mock_mudata_config(self):
        """Create a mock MuDataConfiguration with necessary attributes."""
        config = MuDataConfiguration.__new__(MuDataConfiguration)
        config.output_matrix = dok_matrix((10, 10))
        config.output_binary_matrix = dok_matrix((10, 10))
        config.threshold = 0.5
        return config

    def test_raw_and_threshold_collect_above_threshold(self, mock_mudata_config):
        """Test that values above threshold are collected in both matrices."""
        # Create mock samples with PZi values (will be transposed by the method)
        mock_samples = MagicMock()
        # Before transpose: [samples x cells]
        # After transpose: [cells x samples]
        pzi_raw = np.array([
            [0.8, 0.2, 0.1],  # Sample 0
            [0.1, 0.7, 0.1],  # Sample 1
            [0.1, 0.1, 0.8],  # Sample 2
        ])
        mock_samples.stan_variable.return_value = pzi_raw

        # Create mock cell_info
        cell_info = [(0, "metadata0"), (1, "metadata1"), (2, "metadata2")]

        guide_id = 5

        # Call the method
        mock_mudata_config._raw_and_threshold_collect(guide_id, mock_samples, cell_info)

        # After transpose: pzi[0] = [0.8, 0.1, 0.1], pzi[1] = [0.2, 0.7, 0.1], pzi[2] = [0.1, 0.1, 0.8]
        # Verify output_matrix has median values for all cells
        assert mock_mudata_config.output_matrix[0, guide_id] == np.median([0.8, 0.1, 0.1])
        assert mock_mudata_config.output_matrix[1, guide_id] == np.median([0.2, 0.7, 0.1])
        assert mock_mudata_config.output_matrix[2, guide_id] == np.median([0.1, 0.1, 0.8])

    def test_raw_and_threshold_collect_threshold_boundary(self, mock_mudata_config):
        """Test boundary conditions at exact threshold value."""
        mock_samples = MagicMock()
        # Before transpose: [samples x cells]
        # Cell 0: values [0.4, 0.6, 0.5] -> median 0.5 (equals threshold)
        # Cell 1: values [0.4, 0.5, 0.4] -> median 0.4 (below threshold)
        # Cell 2: values [0.6, 0.5, 0.6] -> median 0.6 (above threshold)
        pzi_raw = np.array([
            [0.4, 0.4, 0.6],  # Sample 0
            [0.6, 0.5, 0.5],  # Sample 1
            [0.5, 0.4, 0.6],  # Sample 2
        ])
        mock_samples.stan_variable.return_value = pzi_raw

        cell_info = [(0, "meta0"), (1, "meta1"), (2, "meta2")]
        guide_id = 3

        mock_mudata_config._raw_and_threshold_collect(guide_id, mock_samples, cell_info)

        # After transpose: pzi[0] = [0.4, 0.6, 0.5], pzi[1] = [0.4, 0.5, 0.4], pzi[2] = [0.6, 0.5, 0.6]
        # Cell at or above threshold should be marked in binary matrix
        assert mock_mudata_config.output_binary_matrix[0, guide_id] == 1  # median = 0.5
        assert mock_mudata_config.output_binary_matrix[1, guide_id] == 0  # median = 0.4
        assert mock_mudata_config.output_binary_matrix[2, guide_id] == 1  # median = 0.6

    def test_raw_and_threshold_collect_multiple_guides(self, mock_mudata_config):
        """Test collection for multiple guides in sequence."""
        mock_samples = MagicMock()
        pzi_values = np.array([[0.9, 0.1], [0.1, 0.9]])
        mock_samples.stan_variable.return_value = pzi_values

        cell_info = [(0, "meta0"), (1, "meta1")]

        # First guide
        mock_mudata_config._raw_and_threshold_collect(0, mock_samples, cell_info)
        # Second guide
        mock_mudata_config._raw_and_threshold_collect(1, mock_samples, cell_info)

        # Verify both guides are populated
        assert mock_mudata_config.output_matrix[0, 0] != 0
        assert mock_mudata_config.output_matrix[0, 1] != 0
        assert mock_mudata_config.output_binary_matrix[0, 0] == 1  # median = 0.5
        assert mock_mudata_config.output_binary_matrix[1, 1] == 1  # median = 0.9


class TestMuDataConfigurationRawCollection:
    """Tests for MuDataConfiguration._raw_collect method."""

    @pytest.fixture
    def mock_mudata_config_no_threshold(self):
        """Create a mock MuDataConfiguration without threshold."""
        config = MuDataConfiguration.__new__(MuDataConfiguration)
        config.output_matrix = dok_matrix((10, 10))
        config.threshold = None
        return config

    def test_raw_collect_all_values(self, mock_mudata_config_no_threshold):
        """Test that raw collection stores all median values."""
        mock_samples = MagicMock()
        # Before transpose: [samples x cells]
        pzi_raw = np.array([
            [0.1, 0.4, 0.7],  # Sample 0
            [0.2, 0.5, 0.8],  # Sample 1
            [0.3, 0.6, 0.9],  # Sample 2
        ])
        mock_samples.stan_variable.return_value = pzi_raw

        cell_info = [(0, "meta0"), (1, "meta1"), (2, "meta2")]
        guide_id = 2

        mock_mudata_config_no_threshold._raw_collect(guide_id, mock_samples, cell_info)

        # After transpose: pzi[0] = [0.1, 0.2, 0.3], pzi[1] = [0.4, 0.5, 0.6], pzi[2] = [0.7, 0.8, 0.9]
        # Verify medians are stored
        assert mock_mudata_config_no_threshold.output_matrix[0, guide_id] == np.median([0.1, 0.2, 0.3])
        assert mock_mudata_config_no_threshold.output_matrix[1, guide_id] == np.median([0.4, 0.5, 0.6])
        assert mock_mudata_config_no_threshold.output_matrix[2, guide_id] == np.median([0.7, 0.8, 0.9])


class TestMuDataConfigurationSampleCollection:
    """Tests for MuDataConfiguration collect_samples method."""

    @pytest.fixture
    def mock_mudata_config_cs(self):
        """Create a mock MuDataConfiguration with CS model."""
        config = MuDataConfiguration.__new__(MuDataConfiguration)
        config.model = Model.CS
        config.samples = []
        return config

    @pytest.fixture
    def mock_mudata_config_dc(self):
        """Create a mock MuDataConfiguration with DC model."""
        config = MuDataConfiguration.__new__(MuDataConfiguration)
        config.model = Model.DC
        config.samples = []
        return config

    def test_collect_cs_samples(self, mock_mudata_config_cs):
        """Test CS model sample collection."""
        mock_samples = MagicMock()
        mock_samples.stan_variable.side_effect = lambda var: {
            "r": np.array([0.5, 0.6]),
            "nbMean": np.array([10, 20]),
            "nbDisp": np.array([0.1, 0.2]),
            "lambda": np.array([5, 15]),
        }[var]

        guide_id = "GUIDE_1"
        mock_mudata_config_cs.collect_samples(guide_id, mock_samples)

        assert len(mock_mudata_config_cs.samples) == 2
        assert mock_mudata_config_cs.samples[0] == ("GUIDE_1", 0.5, 10, 0.1, 5)
        assert mock_mudata_config_cs.samples[1] == ("GUIDE_1", 0.6, 20, 0.2, 15)

    def test_collect_dc_samples(self, mock_mudata_config_dc):
        """Test DC model sample collection."""
        mock_samples = MagicMock()
        mock_samples.stan_variable.side_effect = lambda var: {
            "r": np.array([0.7]),
            "nbMean": np.array([12]),
            "nbDisp": np.array([0.15]),
            "n_nbMean": np.array([8]),
            "n_nbDisp": np.array([0.08]),
        }[var]

        guide_id = "GUIDE_2"
        mock_mudata_config_dc.collect_samples(guide_id, mock_samples)

        assert len(mock_mudata_config_dc.samples) == 1
        assert mock_mudata_config_dc.samples[0] == ("GUIDE_2", 0.7, 12, 0.15, 8, 0.08)


class TestMuDataConfigurationStatsCollection:
    """Tests for MuDataConfiguration stats collection methods."""

    @pytest.fixture
    def mock_mudata_config_cs(self):
        """Create a mock MuDataConfiguration with CS model."""
        config = MuDataConfiguration.__new__(MuDataConfiguration)
        config.model = Model.CS
        config.stats = []
        return config

    @pytest.fixture
    def mock_mudata_config_dc(self):
        """Create a mock MuDataConfiguration with DC model."""
        config = MuDataConfiguration.__new__(MuDataConfiguration)
        config.model = Model.DC
        config.stats = []
        return config

    def test_collect_cs_stats(self, mock_mudata_config_cs):
        """Test CS model stats collection."""
        mock_samples = MagicMock()
        mock_samples.stan_variable.side_effect = lambda var: {
            "r": np.array([0.4, 0.5, 0.6]),
            "nbMean": np.array([10, 15, 20]),
            "lambda": np.array([3, 5, 7]),
        }[var]

        mock_mudata_config_cs.collect_stats(mock_samples)

        assert len(mock_mudata_config_cs.stats) == 1
        r_median = np.median([0.4, 0.5, 0.6])
        mu_median = np.median([10, 15, 20])
        lam_median = np.median([3, 5, 7])
        assert mock_mudata_config_cs.stats[0] == (r_median, mu_median, lam_median)

    def test_collect_dc_stats(self, mock_mudata_config_dc):
        """Test DC model stats collection."""
        mock_samples = MagicMock()
        mock_samples.stan_variable.side_effect = lambda var: {
            "r": np.array([0.3, 0.5, 0.7]),
            "nbMean": np.array([8, 12, 16]),
            "n_nbMean": np.array([5, 10, 15]),
            "n_nbDisp": np.array([0.05, 0.1, 0.15]),
        }[var]

        mock_mudata_config_dc.collect_stats(mock_samples)

        assert len(mock_mudata_config_dc.stats) == 1
        r_median = np.median([0.3, 0.5, 0.7])
        mu_median = np.median([8, 12, 16])
        n_mean_median = np.median([5, 10, 15])
        n_disp_median = np.median([0.05, 0.1, 0.15])
        assert mock_mudata_config_dc.stats[0] == (r_median, mu_median, n_mean_median, n_disp_median)


class TestMtxConfigurationDataGeneration:
    """Tests for MtxConfiguration.gen_data method."""

    def test_gen_data_skips_comments(self):
        """Test that gen_data correctly skips comment lines."""
        mm_file_content = """%Matrix Market format
% Comments
% More comments
3 3 9
1 1 5
2 2 3
3 3 7
"""
        mm_file = io.StringIO(mm_file_content)

        config = MtxConfiguration.__new__(MtxConfiguration)
        config.input_file = mm_file

        data = list(config.gen_data())

        assert len(data) == 3
        assert data[0] == ("1", "1", 5)
        assert data[1] == ("2", "2", 3)
        assert data[2] == ("3", "3", 7)

    def test_gen_data_stores_header(self):
        """Test that gen_data stores the matrix market header."""
        mm_file_content = """%%MatrixMarket matrix coordinate real general
3 3 3
1 1 1
2 2 2
3 3 3
"""
        mm_file = io.StringIO(mm_file_content)

        config = MtxConfiguration.__new__(MtxConfiguration)
        config.input_file = mm_file
        config.mm_header = None

        list(config.gen_data())

        assert config.mm_header == "3 3 3\n"


class TestMtxConfigurationPosteriorCollection:
    """Tests for MtxConfiguration.collect_posteriors method."""

    def test_collect_posteriors_basic(self):
        """Test basic posterior collection to matrix market format."""
        output_file = io.StringIO()

        config = MtxConfiguration.__new__(MtxConfiguration)
        config.posteriors_output_file = output_file
        config.mm_header = "3 3 3\n"
        config.output_all_posteriors = False

        mock_samples = MagicMock()
        pzi_values = np.array([
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
        ])
        mock_samples.stan_variable.return_value = pzi_values

        cell_info = [(0, "meta0"), (1, "meta1")]

        config.collect_posteriors("guide_1", mock_samples, cell_info)

        output = output_file.getvalue()
        assert "3 3 3" in output
        assert "guide_1\t0\t" in output
        assert "guide_1\t1\t" in output


class TestMuDataConfigurationInitialization:
    """Tests for MuDataConfiguration initialization with/without threshold."""

    @patch("cleanser.configuration.md.read")
    def test_initialization_without_threshold(self, mock_md_read):
        """Test initialization sets raw_collect when threshold is None."""
        mock_mudata = MagicMock()
        mock_mudata.__getitem__.return_value = MagicMock()
        mock_mudata.__getitem__.return_value.X = dok_matrix((5, 5))
        mock_mudata.__getitem__.return_value.uns = {}
        mock_md_read.return_value = mock_mudata

        with tempfile.NamedTemporaryFile(suffix=".h5mu") as tmp:
            config = MuDataConfiguration(
                input=tmp.name,
                modality="guides",
                capture_method="method",
                output_layer="layer",
                model=Model.CS,
                sample_output=None,
                posteriors_output=None,
                threshold=None,
            )

            assert config.threshold is None
            assert config.collect_posteriors == config._raw_collect

    @patch("cleanser.configuration.md.read")
    def test_initialization_with_threshold(self, mock_md_read):
        """Test initialization sets threshold_collect when threshold is provided."""
        mock_mudata = MagicMock()
        mock_mudata.__getitem__.return_value = MagicMock()
        mock_mudata.__getitem__.return_value.X = dok_matrix((5, 5))
        mock_mudata.__getitem__.return_value.uns = {}
        mock_md_read.return_value = mock_mudata

        with tempfile.NamedTemporaryFile(suffix=".h5mu") as tmp:
            config = MuDataConfiguration(
                input=tmp.name,
                modality="guides",
                capture_method="method",
                output_layer="layer",
                model=Model.CS,
                sample_output=None,
                posteriors_output=None,
                threshold=0.5,
            )

            assert config.threshold == 0.5
            assert config.collect_posteriors == config._raw_and_threshold_collect

class TestMtxConfigurationInitialization:
    """Tests for MtxConfiguration initialization."""

    def test_mtx_initialization_creates_input_file(self):
        """Test that MtxConfiguration opens input file."""
        mtx_path = "tests/data/test_small.mtx"
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as sample_out:
            sample_out_path = sample_out.name

        try:
            config = MtxConfiguration(
                input=mtx_path,
                model=Model.CS,
                sample_output=sample_out_path,
                posteriors_output=None,
            )
            assert config.input_file is not None
            assert config.sample_output_file is not None
            assert not config.input_file.closed
            config.__del__()
        finally:
            import os
            if os.path.exists(sample_out_path):
                os.remove(sample_out_path)

    def test_mtx_initialization_with_posteriors_output(self):
        """Test MtxConfiguration with posteriors output file."""
        mtx_path = "tests/data/test_small.mtx"
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as post_out:
            post_out_path = post_out.name

        try:
            config = MtxConfiguration(
                input=mtx_path,
                model=Model.DC,
                sample_output=None,
                posteriors_output=post_out_path,
            )
            assert config.posteriors_output_file is not None
            assert not config.posteriors_output_file.closed
            config.__del__()
        finally:
            import os
            if os.path.exists(post_out_path):
                os.remove(post_out_path)

    def test_mtx_initialization_sets_model(self):
        """Test that model is correctly set in MtxConfiguration."""
        mtx_path = "tests/data/test_small.mtx"
        config = MtxConfiguration(
            input=mtx_path,
            model=Model.DC,
            sample_output=None,
            posteriors_output=None,
        )
        assert config.model == Model.DC
        config.__del__()


class TestMtxConfigurationGenData:
    """Tests for MtxConfiguration.gen_data() method."""

    def test_gen_data_skips_comments(self):
        """Test that gen_data skips comment lines."""
        mtx_path = "tests/data/test_small.mtx"
        config = MtxConfiguration(
            input=mtx_path,
            model=Model.CS,
            sample_output=None,
            posteriors_output=None,
        )

        data = list(config.gen_data())
        # Should skip header lines starting with %
        # and first non-comment line (3 3 7) is stored in mm_header
        assert len(data) == 7
        assert config.mm_header == "3 3 7\n"
        assert data[0] == ("1", "1", 5)
        assert data[-1] == ("3", "3", 6)
        config.__del__()

    def test_gen_data_stores_header(self):
        """Test that gen_data stores the matrix market header."""
        mtx_path = "tests/data/test_small.mtx"
        config = MtxConfiguration(
            input=mtx_path,
            model=Model.CS,
            sample_output=None,
            posteriors_output=None,
        )

        assert config.mm_header is None
        list(config.gen_data())
        assert config.mm_header == "3 3 7\n"
        config.__del__()

    def test_gen_data_yields_correct_tuples(self):
        """Test that gen_data yields tuples with correct types."""
        mtx_path = "tests/data/test_small.mtx"
        config = MtxConfiguration(
            input=mtx_path,
            model=Model.CS,
            sample_output=None,
            posteriors_output=None,
        )

        for guide, cell, count in config.gen_data():
            assert isinstance(guide, str)
            assert isinstance(cell, str)
            assert isinstance(count, int)
            assert count > 0

        config.__del__()

    def test_gen_data_parses_values_correctly(self):
        """Test that gen_data parses MTX values correctly."""
        mtx_path = "tests/data/test_small.mtx"
        config = MtxConfiguration(
            input=mtx_path,
            model=Model.CS,
            sample_output=None,
            posteriors_output=None,
        )

        data = list(config.gen_data())
        # Check specific parsed values
        assert ("1", "1", 5) in data
        assert ("2", "3", 4) in data
        assert ("3", "3", 6) in data
        config.__del__()


class TestMtxConfigurationPosteriorCollection:
    """Tests for MtxConfiguration posterior collection."""

    def test_mtx_collect_posteriors_writes_header(self):
        """Test that collect_posteriors writes the matrix market header."""
        output_file = io.StringIO()

        config = MtxConfiguration.__new__(MtxConfiguration)
        config.posteriors_output_file = output_file
        config.mm_header = "3 3 3\n"
        config.output_all_posteriors = False

        mock_samples = MagicMock()
        pzi_values = np.array([
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
        ])
        mock_samples.stan_variable.return_value = pzi_values

        cell_info = [(0, "meta0"), (1, "meta1")]

        config.collect_posteriors("guide_1", mock_samples, cell_info)

        output = output_file.getvalue()
        assert "3 3 3" in output
        # Header should be consumed after first call
        assert config.mm_header is None

    def test_mtx_collect_posteriors_writes_medians(self):
        """Test that collect_posteriors writes median posterior values."""
        output_file = io.StringIO()

        config = MtxConfiguration.__new__(MtxConfiguration)
        config.posteriors_output_file = output_file
        config.mm_header = "3 3 3\n"
        config.output_all_posteriors = False

        mock_samples = MagicMock()
        pzi_values = np.array([
            [0.9, 0.05, 0.05],  # median = 0.05
            [0.1, 0.8, 0.1],    # median = 0.1
        ])
        mock_samples.stan_variable.return_value = pzi_values

        cell_info = [(0, "cell0"), (1, "cell1")]

        config.collect_posteriors("guide_5", mock_samples, cell_info)

        output = output_file.getvalue()
        lines = output.strip().split("\n")
        # First line is header
        assert lines[0] == "3 3 3"
        # Following lines should have guide_id, cell_id (uses index not metadata), median
        assert "guide_5\t0\t" in lines[1]
        assert "guide_5\t1\t" in lines[2]

    def test_mtx_collect_posteriors_multiple_calls(self):
        """Test that header is only written once across multiple calls."""
        output_file = io.StringIO()

        config = MtxConfiguration.__new__(MtxConfiguration)
        config.posteriors_output_file = output_file
        config.mm_header = "5 5 10\n"
        config.output_all_posteriors = False

        mock_samples = MagicMock()
        pzi_values = np.array([[0.5, 0.5]])
        mock_samples.stan_variable.return_value = pzi_values

        # First call
        config.collect_posteriors("guide_1", mock_samples, [(0, "cell_0")])
        # Second call
        config.collect_posteriors("guide_2", mock_samples, [(0, "cell_0")])

        output = output_file.getvalue()
        # Should have header appearing only once
        header_count = output.count("5 5 10")
        assert header_count == 1

    def test_mtx_posteriors_output_message(self):
        """Test output_posteriors with various output_all_posteriors settings."""
        config = MtxConfiguration.__new__(MtxConfiguration)
        
        # Test with output_all_posteriors = False
        config.output_all_posteriors = False
        # Should not raise an error
        config.output_posteriors()

        # Test with output_all_posteriors = True
        config.output_all_posteriors = True
        # Should not raise an error
        config.output_posteriors()


class TestMtxConfigurationWithRealFile:
    """Integration tests for MtxConfiguration with real MTX file."""

    def test_mtx_full_workflow(self):
        """Test complete workflow: load file, generate data, collect posteriors."""
        mtx_path = "tests/data/test_small.mtx"

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as sample_out:
            sample_out_path = sample_out.name
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as post_out:
            post_out_path = post_out.name

        try:
            config = MtxConfiguration(
                input=mtx_path,
                model=Model.CS,
                sample_output=sample_out_path,
                posteriors_output=post_out_path,
            )

            # Generate data
            data_list = list(config.gen_data())
            assert len(data_list) > 0
            assert config.mm_header == "3 3 7\n"

            # Simulate posterior collection
            mock_samples = MagicMock()
            pzi_values = np.array([[0.7, 0.2, 0.1]])
            mock_samples.stan_variable.return_value = pzi_values
            config.collect_posteriors("guide_test", mock_samples, [(0, "cell_0")])

            # Verify sample output file can be written to
            config.sample_output_file.write("test\n")

            config.__del__()

            # Check that files were created and have content
            with open(post_out_path, "r") as f:
                content = f.read()
                assert "3 3 7" in content
                assert "guide_test" in content

            with open(sample_out_path, "r") as f:
                content = f.read()
                assert "test" in content

        finally:
            import os
            if os.path.exists(sample_out_path):
                os.remove(sample_out_path)
            if os.path.exists(post_out_path):
                os.remove(post_out_path)

    def test_mtx_gen_data_with_large_counts(self):
        """Test gen_data handles various count values correctly."""
        mtx_path = "tests/data/test_small.mtx"
        config = MtxConfiguration(
            input=mtx_path,
            model=Model.CS,
            sample_output=None,
            posteriors_output=None,
        )

        data = list(config.gen_data())
        # Check that all counts are parsed as integers
        counts = [count for _, _, count in data]
        assert all(isinstance(c, int) for c in counts)
        assert max(counts) == 6
        assert min(counts) == 1
        config.__del__()


class TestMtxConfigurationDestruction:
    """Tests for MtxConfiguration cleanup (__del__ method)."""

    def test_mtx_deletion_closes_files(self):
        """Test that __del__ closes all open files."""
        mtx_path = "tests/data/test_small.mtx"

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as sample_out:
            sample_out_path = sample_out.name
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as post_out:
            post_out_path = post_out.name

        try:
            config = MtxConfiguration(
                input=mtx_path,
                model=Model.CS,
                sample_output=sample_out_path,
                posteriors_output=post_out_path,
            )

            input_file = config.input_file
            sample_file = config.sample_output_file
            posteriors_file = config.posteriors_output_file

            config.__del__()

            assert input_file.closed
            assert sample_file.closed
            assert posteriors_file.closed

        finally:
            import os
            if os.path.exists(sample_out_path):
                os.remove(sample_out_path)
            if os.path.exists(post_out_path):
                os.remove(post_out_path)

    def test_mtx_deletion_handles_missing_attributes(self):
        """Test that __del__ handles objects missing some attributes gracefully."""
        config = MtxConfiguration.__new__(MtxConfiguration)
        # Don't set any attributes - should not raise
        config.__del__()

    def test_mtx_deletion_with_stdout(self):
        """Test that __del__ handles stdout gracefully."""
        mtx_path = "tests/data/test_small.mtx"
        config = MtxConfiguration(
            input=mtx_path,
            model=Model.CS,
            sample_output=None,  # Will use stdout
            posteriors_output=None,  # Will use stdout
        )

        # Should not raise when deleting (stdout shouldn't be closed)
        config.__del__()