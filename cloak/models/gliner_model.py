"""GLiNER model wrapper supporting both ONNX and PyTorch backends."""

import logging
import threading
from pathlib import Path
from typing import Any

from gliner import GLiNER

logger = logging.getLogger(__name__)


class GLiNERModel:
    """Wrapper around GLiNER supporting ONNX and PyTorch inference.

    Args:
        model_path: HuggingFace model ID or local directory path.
        use_onnx: If True, load ONNX-optimized model. If False, use PyTorch.
        onnx_model_file: Filename of the ONNX model inside the model directory.
        device: Device to map model to (default: "cpu").
        local_files_only: If True, only look for local model files.
    """

    def __init__(
        self,
        model_path: str,
        use_onnx: bool = True,
        onnx_model_file: str = "model.onnx",
        device: str = "cpu",
        local_files_only: bool = False,
    ):
        self.model_path = model_path
        self.use_onnx = use_onnx
        self.onnx_model_file = onnx_model_file

        # Validate local path if it's not a HuggingFace model ID
        if not self._is_hf_model_id(model_path):
            path = Path(model_path)
            if not path.exists():
                raise FileNotFoundError(f"Model path does not exist: {model_path}")

        self._lock = threading.Lock()
        self.model = self._load_model(
            model_path, use_onnx, onnx_model_file, device, local_files_only
        )
        logger.info("GLiNERModel loaded: path=%s, onnx=%s, device=%s", model_path, use_onnx, device)

    def _load_model(
        self,
        model_path: str,
        use_onnx: bool,
        onnx_model_file: str,
        device: str = "cpu",
        local_files_only: bool = False,
    ) -> GLiNER:
        return GLiNER.from_pretrained(
            model_path,
            load_onnx_model=use_onnx,
            onnx_model_file=onnx_model_file,
            map_location=device,
            local_files_only=local_files_only,
        )

    def predict_entities(
        self,
        text: str,
        labels: list[str],
        threshold: float = 0.5,
        flat_ner: bool = True,
    ) -> list[dict[str, Any]]:
        """Run NER on a single text. Returns list of entity dicts."""
        if not text or not text.strip():
            return []
        with self._lock:
            return self.model.predict_entities(text, labels, threshold=threshold, flat_ner=flat_ner)

    def batch_predict_entities(
        self,
        texts: list[str],
        labels: list[str],
        threshold: float = 0.5,
        flat_ner: bool = True,
    ) -> list[list[dict[str, Any]]]:
        """Run NER on a batch of texts."""
        if not texts:
            return []
        with self._lock:
            return self.model.batch_predict_entities(
                texts, labels, threshold=threshold, flat_ner=flat_ner
            )

    def get_model_info(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "use_onnx": self.use_onnx,
            "onnx_model_file": self.onnx_model_file,
            "status": "loaded",
            "type": "GLiNER",
        }

    @staticmethod
    def _is_hf_model_id(path: str) -> bool:
        """Check if path looks like a HuggingFace model ID (org/model).

        Returns False for any path that exists on the local filesystem,
        even if it has the org/model two-part structure.
        """
        if Path(path).exists():
            return False
        parts = path.split("/")
        return len(parts) == 2 and not Path(path).is_absolute()
