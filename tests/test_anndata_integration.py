"""Tests for AnnData functionality in CLEANSER."""

import argparse
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import anndata as ad
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from cleanser.configuration import AnnDataConfiguration, Model
from cleanser.run import get_configuration


TEST_DATA_DIR = Path(__file__).parent / "data"
TEST_ANNDATA_FILE = TEST_DATA_DIR / "test_anndata_small.h5ad"


@pytest.fixture
def test_anndata():
    if not TEST_ANNDATA_FILE.exists():
        pytest.skip(f"Test data not found: {TEST_ANNDATA_FILE}")
    return ad.read_h5ad(str(TEST_ANNDATA_FILE))


def _make_args(input_file, output_file, *, dc=False, cs=False, capture_method_key=None,
               output_layer="guide_assignment", threshold=None, modality=None):
    """Build a minimal args namespace for get_configuration."""
    args = argparse.Namespace(
        input=input_file,
        dc=dc,
        cs=cs,
        capture_method_key=capture_method_key,
        modality=modality,
        output_layer=output_layer,
        posteriors_output=output_file,
        so=None,
        threshold=threshold,
    )
    return args


class TestAnnDataFile:
    def test_file_exists(self):
        assert TEST_ANNDATA_FILE.exists(), f"Test data not found: {TEST_ANNDATA_FILE}"

    def test_file_size(self):
        assert TEST_ANNDATA_FILE.stat().st_size < 500 * 1024

    def test_file_loads(self, test_anndata):
        assert test_anndata is not None
        assert test_anndata.shape[0] > 0
        assert test_anndata.shape[1] > 0
        assert hasattr(test_anndata, "X")


class TestAnnDataConfigurationCreation:
    def test_basic_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AnnDataConfiguration(
                input=str(TEST_ANNDATA_FILE),
                capture_method=None,
                output_layer="guide_assignment",
                model=Model.CS,
                sample_output=None,
                posteriors_output=str(Path(tmpdir) / "out.h5ad"),
                threshold=None,
            )
            assert config.model == Model.CS
            assert config.output_layer == "guide_assignment"

    def test_dc_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AnnDataConfiguration(
                input=str(TEST_ANNDATA_FILE),
                capture_method=None,
                output_layer="guide_assignment",
                model=Model.DC,
                sample_output=None,
                posteriors_output=str(Path(tmpdir) / "out.h5ad"),
                threshold=None,
            )
            assert config.model == Model.DC

    def test_output_matrices_initialized(self):
        config = AnnDataConfiguration(
            input=str(TEST_ANNDATA_FILE),
            capture_method=None,
            output_layer="guide_assignment",
            model=Model.CS,
            sample_output=None,
            posteriors_output=None,
            threshold=0.5,
        )
        assert config.output_matrix is not None
        assert config.output_binary_matrix is not None
        assert config.output_matrix.shape == config.output_binary_matrix.shape

    def test_shape_matches_input(self, test_anndata):
        config = AnnDataConfiguration(
            input=str(TEST_ANNDATA_FILE),
            capture_method=None,
            output_layer="guide_assignment",
            model=Model.CS,
            sample_output=None,
            posteriors_output=None,
            threshold=None,
        )
        assert config.output_matrix.shape == test_anndata.shape


class TestAnnDataConfigurationThreshold:
    def test_without_threshold_uses_raw_collect(self):
        config = AnnDataConfiguration(
            input=str(TEST_ANNDATA_FILE),
            capture_method=None,
            output_layer="guide_assignment",
            model=Model.CS,
            sample_output=None,
            posteriors_output=None,
            threshold=None,
        )
        assert config.threshold is None
        assert config.collect_posteriors == config._raw_collect

    def test_with_threshold_uses_threshold_collect(self):
        config = AnnDataConfiguration(
            input=str(TEST_ANNDATA_FILE),
            capture_method=None,
            output_layer="guide_assignment",
            model=Model.CS,
            sample_output=None,
            posteriors_output=None,
            threshold=0.5,
        )
        assert config.threshold == 0.5
        assert config.collect_posteriors == config._raw_and_threshold_collect

    def test_threshold_variation(self):
        for threshold in [None, 0.1, 0.5, 0.9, 1.0]:
            config = AnnDataConfiguration(
                input=str(TEST_ANNDATA_FILE),
                capture_method=None,
                output_layer="guide_assignment",
                model=Model.CS,
                sample_output=None,
                posteriors_output=None,
                threshold=threshold,
            )
            assert config.threshold == threshold


class TestAnnDataConfigurationCaptureMethod:
    def test_capture_method_cs_from_uns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adata = ad.read_h5ad(str(TEST_ANNDATA_FILE))
            adata.uns["capture_type"] = ["CROP-seq"]
            h5ad_path = str(Path(tmpdir) / "capture_cs.h5ad")
            adata.write_h5ad(h5ad_path)

            config = AnnDataConfiguration(
                input=h5ad_path,
                capture_method="capture_type",
                output_layer="guide_assignment",
                model=None,
                sample_output=None,
                posteriors_output=None,
                threshold=None,
            )
            assert config.model == Model.CS

    def test_capture_method_dc_from_uns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adata = ad.read_h5ad(str(TEST_ANNDATA_FILE))
            adata.uns["capture_type"] = ["direct capture"]
            h5ad_path = str(Path(tmpdir) / "capture_dc.h5ad")
            adata.write_h5ad(h5ad_path)

            config = AnnDataConfiguration(
                input=h5ad_path,
                capture_method="capture_type",
                output_layer="guide_assignment",
                model=None,
                sample_output=None,
                posteriors_output=None,
                threshold=None,
            )
            assert config.model == Model.DC

    def test_explicit_model_overrides_uns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adata = ad.read_h5ad(str(TEST_ANNDATA_FILE))
            adata.uns["capture_type"] = ["CROP-seq"]
            h5ad_path = str(Path(tmpdir) / "capture_override.h5ad")
            adata.write_h5ad(h5ad_path)

            config = AnnDataConfiguration(
                input=h5ad_path,
                capture_method=None,
                output_layer="guide_assignment",
                model=Model.DC,
                sample_output=None,
                posteriors_output=None,
                threshold=None,
            )
            assert config.model == Model.DC


class TestAnnDataGenData:
    def test_gen_data_yields_entries(self):
        config = AnnDataConfiguration(
            input=str(TEST_ANNDATA_FILE),
            capture_method=None,
            output_layer="guide_assignment",
            model=Model.CS,
            sample_output=None,
            posteriors_output=None,
            threshold=None,
        )
        data = list(config.gen_data())
        assert len(data) > 0

    def test_gen_data_entry_shape(self):
        config = AnnDataConfiguration(
            input=str(TEST_ANNDATA_FILE),
            capture_method=None,
            output_layer="guide_assignment",
            model=Model.CS,
            sample_output=None,
            posteriors_output=None,
            threshold=None,
        )
        for guide_id, cell_id, count in config.gen_data():
            assert isinstance(guide_id, (int, np.integer))
            assert isinstance(cell_id, (int, np.integer))
            assert isinstance(count, (int, float, np.integer, np.floating))
            assert count > 0

    def test_gen_data_consistent_with_input(self, test_anndata):
        config = AnnDataConfiguration(
            input=str(TEST_ANNDATA_FILE),
            capture_method=None,
            output_layer="guide_assignment",
            model=Model.CS,
            sample_output=None,
            posteriors_output=None,
            threshold=None,
        )
        data = list(config.gen_data())
        total_count = sum(c for _, _, c in data)
        assert total_count == pytest.approx(test_anndata.X.sum())


class TestAnnDataOutputPosteriors:
    def _make_mock_samples(self, n_cells, value=0.8):
        pzi_raw = np.full((1, n_cells), value)
        mock = MagicMock()
        mock.stan_variable.return_value = pzi_raw
        return mock

    def test_output_writes_h5ad(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "out.h5ad"
            config = AnnDataConfiguration(
                input=str(TEST_ANNDATA_FILE),
                capture_method=None,
                output_layer="guide_assignment",
                model=Model.CS,
                sample_output=None,
                posteriors_output=str(out_path),
                threshold=None,
            )
            config.output_posteriors()
            assert out_path.exists()

    def test_output_layer_present_without_threshold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "out.h5ad"
            config = AnnDataConfiguration(
                input=str(TEST_ANNDATA_FILE),
                capture_method=None,
                output_layer="guide_assignment",
                model=Model.CS,
                sample_output=None,
                posteriors_output=str(out_path),
                threshold=None,
            )
            mock_samples = self._make_mock_samples(config.guides.shape[0])
            cell_info = [(i, None) for i in range(config.guides.shape[0])]
            config.collect_posteriors(0, mock_samples, cell_info)
            config.output_posteriors()

            result = ad.read_h5ad(str(out_path))
            assert "guide_assignment" in result.layers

    def test_output_two_layers_with_threshold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "out.h5ad"
            config = AnnDataConfiguration(
                input=str(TEST_ANNDATA_FILE),
                capture_method=None,
                output_layer="guide_assignment",
                model=Model.CS,
                sample_output=None,
                posteriors_output=str(out_path),
                threshold=0.5,
            )
            mock_samples = self._make_mock_samples(config.guides.shape[0])
            cell_info = [(i, None) for i in range(config.guides.shape[0])]
            config.collect_posteriors(0, mock_samples, cell_info)
            config.output_posteriors()

            result = ad.read_h5ad(str(out_path))
            assert "guide_assignment" in result.layers
            assert "guide_assignment_posteriors" in result.layers

    def test_threshold_binarizes_assignments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "out.h5ad"
            config = AnnDataConfiguration(
                input=str(TEST_ANNDATA_FILE),
                capture_method=None,
                output_layer="guide_assignment",
                model=Model.CS,
                sample_output=None,
                posteriors_output=str(out_path),
                threshold=0.5,
            )
            # All posteriors = 0.8, above threshold — expect all 1s in binary layer
            mock_samples = self._make_mock_samples(config.guides.shape[0], value=0.8)
            cell_info = [(i, None) for i in range(config.guides.shape[0])]
            config.collect_posteriors(0, mock_samples, cell_info)
            config.output_posteriors()

            result = ad.read_h5ad(str(out_path))
            binary = result.layers["guide_assignment"].toarray()
            assert np.all(binary[:, 0] == 1)

    def test_threshold_excludes_below_threshold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "out.h5ad"
            config = AnnDataConfiguration(
                input=str(TEST_ANNDATA_FILE),
                capture_method=None,
                output_layer="guide_assignment",
                model=Model.CS,
                sample_output=None,
                posteriors_output=str(out_path),
                threshold=0.9,
            )
            # All posteriors = 0.3, below threshold — expect all 0s in binary layer
            mock_samples = self._make_mock_samples(config.guides.shape[0], value=0.3)
            cell_info = [(i, None) for i in range(config.guides.shape[0])]
            config.collect_posteriors(0, mock_samples, cell_info)
            config.output_posteriors()

            result = ad.read_h5ad(str(out_path))
            binary = result.layers["guide_assignment"].toarray()
            assert np.all(binary == 0)


class TestGetConfigurationRouting:
    def test_h5ad_routes_to_anndata_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _make_args(
                str(TEST_ANNDATA_FILE),
                str(Path(tmpdir) / "out.h5ad"),
                cs=True,
            )
            config = get_configuration(args)
            assert isinstance(config, AnnDataConfiguration)

    def test_h5ad_does_not_require_modality(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _make_args(
                str(TEST_ANNDATA_FILE),
                str(Path(tmpdir) / "out.h5ad"),
                cs=True,
                modality=None,
            )
            config = get_configuration(args)
            assert isinstance(config, AnnDataConfiguration)

    def test_h5ad_requires_output_layer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _make_args(
                str(TEST_ANNDATA_FILE),
                str(Path(tmpdir) / "out.h5ad"),
                cs=True,
                output_layer=None,
            )
            with pytest.raises(argparse.ArgumentError, match="--output-layer"):
                get_configuration(args)

    def test_h5ad_requires_posteriors_output(self):
        args = _make_args(
            str(TEST_ANNDATA_FILE),
            None,
            cs=True,
        )
        with pytest.raises(argparse.ArgumentError, match="--posteriors-output"):
            get_configuration(args)

    def test_h5ad_requires_model_or_capture_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _make_args(
                str(TEST_ANNDATA_FILE),
                str(Path(tmpdir) / "out.h5ad"),
                dc=False,
                cs=False,
                capture_method_key=None,
            )
            with pytest.raises(argparse.ArgumentError, match="capture method"):
                get_configuration(args)

    def test_h5ad_accepts_capture_method_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _make_args(
                str(TEST_ANNDATA_FILE),
                str(Path(tmpdir) / "out.h5ad"),
                capture_method_key="capture_method",
            )
            config = get_configuration(args)
            assert isinstance(config, AnnDataConfiguration)
