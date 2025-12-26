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
