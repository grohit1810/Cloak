"""Tests for the public API module."""

from unittest.mock import MagicMock, patch

import pytest

import cloak
from cloak.api import _reset_global_instances


class TestGlobalInstances:
    def setup_method(self):
        _reset_global_instances()

    def teardown_method(self):
        _reset_global_instances()

    @patch("cloak.api.CloakExtraction")
    def test_extract_creates_instance_with_model_path(self, mock_cls):
        mock_instance = MagicMock()
        mock_instance.extract_entities.return_value = {"entities": [], "processing_info": {}}
        mock_cls.return_value = mock_instance
        cloak.extract("test", labels=["person"], model_path="/some/path")
        mock_cls.assert_called_once()
        assert mock_cls.call_args[1]["model_path"] == "/some/path"

    @patch("cloak.api.CloakExtraction")
    def test_different_model_path_creates_new_instance(self, mock_cls):
        mock_instance = MagicMock()
        mock_instance.extract_entities.return_value = {"entities": [], "processing_info": {}}
        mock_cls.return_value = mock_instance
        cloak.extract("test", labels=["person"], model_path="/path/a")
        cloak.extract("test", labels=["person"], model_path="/path/b")
        assert mock_cls.call_count == 2

    @patch("cloak.api.CloakExtraction")
    def test_same_model_path_reuses_instance(self, mock_cls):
        mock_instance = MagicMock()
        mock_instance.extract_entities.return_value = {"entities": [], "processing_info": {}}
        mock_cls.return_value = mock_instance
        cloak.extract("test", labels=["person"], model_path="/path/a")
        cloak.extract("test", labels=["person"], model_path="/path/a")
        assert mock_cls.call_count == 1


class TestRedact:
    def setup_method(self):
        _reset_global_instances()

    def teardown_method(self):
        _reset_global_instances()

    @patch("cloak.api.CloakExtraction")
    def test_redact_empty_entities_returns_original(self, mock_cls):
        mock_instance = MagicMock()
        mock_instance.extract_entities.return_value = {"entities": [], "processing_info": {}}
        mock_cls.return_value = mock_instance
        result = cloak.redact("Hello world", model_path="/fake")
        assert result["anonymized_text"] == "Hello world"


class TestReplace:
    def setup_method(self):
        _reset_global_instances()

    def teardown_method(self):
        _reset_global_instances()

    @patch("cloak.api.CloakExtraction")
    def test_replace_empty_entities_returns_original(self, mock_cls):
        mock_instance = MagicMock()
        mock_instance.extract_entities.return_value = {"entities": [], "processing_info": {}}
        mock_cls.return_value = mock_instance
        result = cloak.replace("Hello world", model_path="/fake")
        assert result["anonymized_text"] == "Hello world"


class TestReplaceWithData:
    def setup_method(self):
        _reset_global_instances()

    def teardown_method(self):
        _reset_global_instances()

    def test_replace_with_data_requires_user_replacements(self):
        with pytest.raises(ValueError, match="user_replacements"):
            cloak.replace_with_data("Hello", model_path="/fake")
