"""Tests for GLiNER model wrapper."""

import os
import threading
from unittest.mock import MagicMock, patch

import pytest

from cloak.models.gliner_model import DEFAULT_MODEL_ID, GLiNERModel


class TestGLiNERModel:
    def test_init_requires_valid_local_path(self):
        with pytest.raises(FileNotFoundError):
            GLiNERModel(model_path="/nonexistent/path")

    @patch("cloak.models.gliner_model.GLiNER")
    def test_init_pytorch_loads_from_hf(self, mock_gliner_cls):
        mock_gliner_cls.from_pretrained.return_value = MagicMock()
        GLiNERModel(model_path="some-org/some-model", use_onnx=False)
        mock_gliner_cls.from_pretrained.assert_called_once_with(
            "some-org/some-model",
            load_onnx_model=False,
            map_location="cpu",
            local_files_only=False,
        )

    @patch("cloak.models.gliner_model.GLiNER")
    def test_init_default_model_path(self, mock_gliner_cls):
        """When model_path is None, should use DEFAULT_MODEL_ID."""
        mock_gliner_cls.from_pretrained.return_value = MagicMock()
        model = GLiNERModel(model_path=None, use_onnx=False)
        assert model.model_path == DEFAULT_MODEL_ID
        mock_gliner_cls.from_pretrained.assert_called_once_with(
            DEFAULT_MODEL_ID,
            load_onnx_model=False,
            map_location="cpu",
            local_files_only=False,
        )

    @patch("cloak.models.gliner_model.GLiNER")
    def test_onnx_auto_export_when_no_cache(self, mock_gliner_cls, tmp_path):
        """When use_onnx=True and no cached ONNX, should download PyTorch and export."""
        mock_model = MagicMock()
        mock_gliner_cls.from_pretrained.return_value = mock_model

        # Make export_to_onnx actually create the file (the real one does)
        tmp_path / "some-org--some-model"

        def fake_export(save_dir, onnx_filename, quantize):
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, onnx_filename), "w") as f:
                f.write("fake onnx")

        mock_model.export_to_onnx.side_effect = fake_export

        with patch("cloak.models.gliner_model._CLOAK_CACHE_DIR", tmp_path):
            GLiNERModel(model_path="some-org/some-model", use_onnx=True)

        # Should have called from_pretrained twice:
        # 1. PyTorch download (load_onnx_model=False)
        # 2. ONNX load after export (load_onnx_model=True)
        calls = mock_gliner_cls.from_pretrained.call_args_list
        assert len(calls) == 2
        assert calls[0][1]["load_onnx_model"] is False
        assert calls[1][1]["load_onnx_model"] is True
        mock_model.export_to_onnx.assert_called_once()

    @patch("cloak.models.gliner_model.GLiNER")
    def test_onnx_loads_from_cache_when_available(self, mock_gliner_cls, tmp_path):
        """When cached ONNX model exists, should load directly without PyTorch."""
        mock_gliner_cls.from_pretrained.return_value = MagicMock()

        # Create fake cached ONNX file
        cache_dir = tmp_path / "some-org--some-model"
        cache_dir.mkdir()
        (cache_dir / "model.onnx").write_text("fake onnx model")

        with patch("cloak.models.gliner_model._CLOAK_CACHE_DIR", tmp_path):
            GLiNERModel(model_path="some-org/some-model", use_onnx=True)

        # Should only call from_pretrained once (ONNX load, no PyTorch download)
        mock_gliner_cls.from_pretrained.assert_called_once()
        call_kwargs = mock_gliner_cls.from_pretrained.call_args[1]
        assert call_kwargs["load_onnx_model"] is True
        assert call_kwargs["local_files_only"] is True

    @patch("cloak.models.gliner_model.GLiNER")
    def test_onnx_local_files_only_raises_when_no_cache(self, mock_gliner_cls, tmp_path):
        """When local_files_only=True and no ONNX cache, should raise FileNotFoundError."""
        with patch("cloak.models.gliner_model._CLOAK_CACHE_DIR", tmp_path):
            with pytest.raises(FileNotFoundError, match="local_files_only"):
                GLiNERModel(
                    model_path="some-org/some-model",
                    use_onnx=True,
                    local_files_only=True,
                )

    @patch("cloak.models.gliner_model.GLiNER")
    def test_predict_entities_delegates_to_gliner(self, mock_gliner_cls):
        mock_model = MagicMock()
        mock_model.predict_entities.return_value = [
            {"start": 0, "end": 4, "text": "John", "label": "person", "score": 0.95}
        ]
        model = GLiNERModel.__new__(GLiNERModel)
        model.model = mock_model
        model.model_path = "test"
        model.use_onnx = True
        model._lock = threading.Lock()

        result = model.predict_entities("John works at Google", ["person"], threshold=0.5)
        assert len(result) == 1
        assert result[0]["text"] == "John"

    def test_predict_entities_returns_empty_on_empty_text(self):
        model = GLiNERModel.__new__(GLiNERModel)
        model.model = MagicMock()
        result = model.predict_entities("", ["person"], threshold=0.5)
        assert result == []

    def test_predict_entities_returns_empty_on_whitespace(self):
        model = GLiNERModel.__new__(GLiNERModel)
        model.model = MagicMock()
        result = model.predict_entities("   ", ["person"], threshold=0.5)
        assert result == []

    @patch("cloak.models.gliner_model.GLiNER")
    def test_batch_predict_entities_delegates(self, mock_gliner_cls):
        mock_model = MagicMock()
        mock_model.batch_predict_entities.return_value = [
            [{"start": 0, "end": 4, "text": "John", "label": "person", "score": 0.95}],
            [{"start": 0, "end": 5, "text": "Paris", "label": "location", "score": 0.90}],
        ]
        model = GLiNERModel.__new__(GLiNERModel)
        model.model = mock_model
        model._lock = threading.Lock()

        result = model.batch_predict_entities(
            ["John works here", "Paris is beautiful"], ["person", "location"]
        )
        assert len(result) == 2
        assert result[0][0]["text"] == "John"
        assert result[1][0]["text"] == "Paris"

    def test_batch_predict_entities_returns_empty_on_empty_list(self):
        model = GLiNERModel.__new__(GLiNERModel)
        model.model = MagicMock()
        result = model.batch_predict_entities([], ["person"])
        assert result == []

    @patch("cloak.models.gliner_model.GLiNER")
    def test_get_model_info(self, mock_gliner_cls):
        model = GLiNERModel.__new__(GLiNERModel)
        model.model = MagicMock()
        model.model_path = "test-path"
        model.use_onnx = True
        model.onnx_model_file = "model.onnx"
        info = model.get_model_info()
        assert info["model_path"] == "test-path"
        assert info["use_onnx"] is True
        assert info["status"] == "loaded"

    def test_is_hf_model_id(self):
        assert GLiNERModel._is_hf_model_id("urchade/gliner_large-v2.1") is True
        assert GLiNERModel._is_hf_model_id("/absolute/path") is False
        assert GLiNERModel._is_hf_model_id("just-a-name") is False
        assert GLiNERModel._is_hf_model_id("org/model/extra") is False

    def test_is_hf_model_id_rejects_existing_relative_path(self):
        if os.path.exists("cloak/models"):
            assert GLiNERModel._is_hf_model_id("cloak/models") is False
