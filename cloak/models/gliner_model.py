"""GLiNER model wrapper supporting both ONNX and PyTorch backends.

Handles automatic model download from HuggingFace and ONNX export on first run.
"""

import logging
import os
import threading
from pathlib import Path
from typing import Any

from gliner import GLiNER

from ..constants import DEFAULT_MODEL_ID

logger = logging.getLogger(__name__)

# Default cache directory for exported ONNX models
_CLOAK_CACHE_DIR = Path(os.environ.get("CLOAK_CACHE_DIR", Path.home() / ".cache" / "cloak"))


class GLiNERModel:
    """Wrapper around GLiNER supporting ONNX and PyTorch inference.

    On first run with `use_onnx=True`, automatically:
    1. Downloads the PyTorch model from HuggingFace
    2. Exports it to ONNX format
    3. Caches the ONNX model at ~/.cache/cloak/<model_id>/
    4. Loads the ONNX model for inference

    Subsequent runs load directly from the ONNX cache.

    Args:
        model_path: HuggingFace model ID or local directory path.
            If None, uses the default model (urchade/gliner_large-v2.1).
        use_onnx: If True (default), use ONNX backend. Automatically exports
            if ONNX model is not found.
        onnx_model_file: Filename of the ONNX model (default: "model.onnx").
        device: Device to map model to (default: "cpu").
        local_files_only: If True, only look for local model files (no downloads).
    """

    def __init__(
        self,
        model_path: str | None = None,
        use_onnx: bool = True,
        onnx_model_file: str = "model.onnx",
        device: str = "cpu",
        local_files_only: bool = False,
    ):
        self.model_path = model_path or DEFAULT_MODEL_ID
        self.use_onnx = use_onnx
        self.onnx_model_file = onnx_model_file
        self._lock = threading.Lock()

        # Validate local path if it's not a HuggingFace model ID
        if not self._is_hf_model_id(self.model_path):
            path = Path(self.model_path)
            if not path.exists():
                raise FileNotFoundError(f"Model path does not exist: {self.model_path}")

        self.model = self._load_model(device, local_files_only)
        logger.info(
            "GLiNERModel loaded: path=%s, onnx=%s, device=%s",
            self.model_path,
            use_onnx,
            device,
        )

    def _load_model(self, device: str, local_files_only: bool) -> GLiNER:
        """Load the model, auto-exporting to ONNX if needed."""
        if not self.use_onnx:
            # PyTorch path — straightforward download + load
            return GLiNER.from_pretrained(
                self.model_path,
                load_onnx_model=False,
                map_location=device,
                local_files_only=local_files_only,
            )

        # ONNX path — check for cached export first
        onnx_dir = self._get_onnx_cache_dir()
        onnx_path = onnx_dir / self.onnx_model_file

        if onnx_path.exists():
            # Cached ONNX model found — load directly
            logger.info("Loading cached ONNX model from %s", onnx_dir)
            return GLiNER.from_pretrained(
                str(onnx_dir),
                load_onnx_model=True,
                onnx_model_file=self.onnx_model_file,
                map_location=device,
                local_files_only=True,
            )

        # Also check if model_path is a local dir that already has the ONNX file
        if not self._is_hf_model_id(self.model_path):
            local_onnx = Path(self.model_path) / self.onnx_model_file
            if local_onnx.exists():
                logger.info("Loading ONNX model from local path %s", self.model_path)
                return GLiNER.from_pretrained(
                    self.model_path,
                    load_onnx_model=True,
                    onnx_model_file=self.onnx_model_file,
                    map_location=device,
                    local_files_only=True,
                )

        # No cached ONNX — download PyTorch, export, then load ONNX
        if local_files_only:
            raise FileNotFoundError(
                f"No cached ONNX model found at {onnx_dir} and local_files_only=True. "
                f"Run once without local_files_only to download and export the model."
            )

        logger.info("No ONNX model found. Downloading PyTorch model and exporting to ONNX...")
        self._download_and_export_onnx(onnx_dir, device)

        logger.info("Loading freshly exported ONNX model from %s", onnx_dir)
        return GLiNER.from_pretrained(
            str(onnx_dir),
            load_onnx_model=True,
            onnx_model_file=self.onnx_model_file,
            map_location=device,
            local_files_only=True,
        )

    def _download_and_export_onnx(self, onnx_dir: Path, device: str) -> None:
        """Download PyTorch model from HuggingFace and export to ONNX."""
        # Step 1: Download PyTorch model
        logger.info("Downloading PyTorch model: %s", self.model_path)
        pytorch_model = GLiNER.from_pretrained(
            self.model_path,
            load_onnx_model=False,
            map_location=device,
        )

        # Step 2: Export to ONNX
        onnx_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Exporting to ONNX at %s (this may take a minute)...", onnx_dir)
        pytorch_model.export_to_onnx(
            save_dir=str(onnx_dir),
            onnx_filename=self.onnx_model_file,
            quantize=False,
        )

        onnx_path = onnx_dir / self.onnx_model_file
        size_mb = onnx_path.stat().st_size / (1024 * 1024)
        logger.info("ONNX export complete: %s (%.1f MB)", onnx_path, size_mb)

    def _get_onnx_cache_dir(self) -> Path:
        """Get the cache directory for exported ONNX models."""
        # Sanitize model ID for use as directory name
        safe_name = self.model_path.replace("/", "--")
        return _CLOAK_CACHE_DIR / safe_name

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
            return self.model.predict_entities(
                text,
                labels,
                threshold=threshold,
                flat_ner=flat_ner,
            )

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
                texts,
                labels,
                threshold=threshold,
                flat_ner=flat_ner,
            )

    def get_model_info(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "use_onnx": self.use_onnx,
            "onnx_model_file": self.onnx_model_file,
            "onnx_cache_dir": str(self._get_onnx_cache_dir()),
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
