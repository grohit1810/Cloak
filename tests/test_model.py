"""Tests for GLiNER model wrapper."""

import threading
from unittest.mock import MagicMock, patch

import pytest

from cloak.models.gliner_model import GLiNERModel


class TestGLiNERModel:
    def test_init_requires_valid_path(self):
        with pytest.raises(FileNotFoundError):
            GLiNERModel(model_path="/nonexistent/path")

    @patch("cloak.models.gliner_model.GLiNER")
    def test_init_loads_onnx_by_default(self, mock_gliner_cls):
        mock_gliner_cls.from_pretrained.return_value = MagicMock()
        model = GLiNERModel.__new__(GLiNERModel)
        model.model = model._load_model(
            "some-org/some-model", use_onnx=True, onnx_model_file="model.onnx"
        )
        mock_gliner_cls.from_pretrained.assert_called_once_with(
            "some-org/some-model",
            load_onnx_model=True,
            onnx_model_file="model.onnx",
            map_location="cpu",
            local_files_only=False,
        )

    @patch("cloak.models.gliner_model.GLiNER")
    def test_init_loads_pytorch_when_onnx_false(self, mock_gliner_cls):
        mock_gliner_cls.from_pretrained.return_value = MagicMock()
        model = GLiNERModel.__new__(GLiNERModel)
        model.model = model._load_model(
            "some-org/some-model", use_onnx=False, onnx_model_file="model.onnx"
        )
        mock_gliner_cls.from_pretrained.assert_called_once_with(
            "some-org/some-model",
            load_onnx_model=False,
            onnx_model_file="model.onnx",
            map_location="cpu",
            local_files_only=False,
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
        assert info["type"] == "GLiNER"

    def test_is_hf_model_id(self):
        assert GLiNERModel._is_hf_model_id("urchade/gliner_large-v2.1") is True
        assert GLiNERModel._is_hf_model_id("/absolute/path") is False
        assert GLiNERModel._is_hf_model_id("just-a-name") is False
        assert GLiNERModel._is_hf_model_id("org/model/extra") is False
