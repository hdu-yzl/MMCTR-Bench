"""Local, provenance-aware encoder adapters for AntM2C extraction."""

import hashlib
import io
import tarfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from mmctr.core import ContractError


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class CheckpointIdentity:
    """Content identity for a local encoder checkpoint, independent of its path."""

    root: str
    sha256: str
    files: Mapping[str, str]

    def __post_init__(self) -> None:
        if len(self.sha256) != 64 or not self.files:
            raise ContractError("checkpoint identity requires files and a SHA-256 digest")
        if any(len(value) != 64 for value in self.files.values()):
            raise ContractError("checkpoint files require SHA-256 digests")
        object.__setattr__(self, "files", MappingProxyType(dict(self.files)))


def fingerprint_checkpoint(root: Path) -> CheckpointIdentity:
    """Hash every regular file in a local checkpoint directory."""

    root = Path(root).resolve()
    if not root.is_dir():
        raise ContractError("encoder checkpoint directory is missing: {}".format(root))
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    if not paths:
        raise ContractError("encoder checkpoint directory is empty: {}".format(root))
    files: Dict[str, str] = {}
    aggregate = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix()
        digest = _sha256_file(path)
        files[relative] = digest
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    return CheckpointIdentity(str(root), aggregate.hexdigest(), files)


def load_tar_images(archive: Path, member_names: Sequence[str]) -> Mapping[str, Any]:
    """Read selected images in one streaming archive pass without extracting to disk."""

    from PIL import Image

    archive = Path(archive)
    if not archive.is_file():
        raise ContractError("AntM2C image archive is missing: {}".format(archive))
    requested = tuple(str(name) for name in member_names)
    if not requested or len(set(requested)) != len(requested):
        raise ContractError("image archive member names must be non-empty and unique")
    remaining = set(requested)
    images: Dict[str, Any] = {}
    with tarfile.open(archive, mode="r|gz") as stream:
        for member in stream:
            if member.name not in remaining:
                continue
            extracted = stream.extractfile(member)
            if extracted is None:
                raise ContractError("image archive member is not a file: {}".format(member.name))
            with Image.open(io.BytesIO(extracted.read())) as image:
                images[member.name] = image.convert("RGB").copy()
            remaining.remove(member.name)
            if not remaining:
                break
    if remaining:
        raise ContractError("image archive members are missing: {}".format(sorted(remaining)))
    return MappingProxyType(images)


class BertPoolerEncoder:
    """Match the legacy AntM2C BERT pooler-output extraction semantics."""

    def __init__(self, checkpoint: Path, device: str) -> None:
        import torch
        from transformers import BertModel, BertTokenizer  # type: ignore[import-untyped]

        self._torch = torch
        self._device = torch.device(device)
        self._tokenizer = BertTokenizer.from_pretrained(
            str(Path(checkpoint)), local_files_only=True
        )
        self._model = BertModel.from_pretrained(str(Path(checkpoint)), local_files_only=True).to(
            self._device
        )
        self._model.eval()

    def encode(self, values: Sequence[Any]) -> np.ndarray:
        texts = [str(value) for value in values]
        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self._device)
        with self._torch.no_grad():
            output = self._model(**inputs).pooler_output
        return output.detach().cpu().numpy().astype(np.float32, copy=False)


class ChineseClipImageEncoder:
    """Match the legacy unnormalised ChineseCLIP image-feature extraction."""

    def __init__(self, checkpoint: Path, device: str) -> None:
        import torch
        from transformers import (
            ChineseCLIPModel,
            ChineseCLIPProcessor,
        )

        self._torch = torch
        self._device = torch.device(device)
        self._processor = ChineseCLIPProcessor.from_pretrained(
            str(Path(checkpoint)), local_files_only=True
        )
        self._model = ChineseCLIPModel.from_pretrained(
            str(Path(checkpoint)), local_files_only=True
        ).to(self._device)
        self._model.eval()

    def encode(self, values: Sequence[Any]) -> np.ndarray:
        inputs = self._processor(images=list(values), return_tensors="pt").to(self._device)
        with self._torch.no_grad():
            output = self._model.get_image_features(**inputs)
        return output.detach().cpu().numpy().astype(np.float32, copy=False)


__all__ = [
    "BertPoolerEncoder",
    "CheckpointIdentity",
    "ChineseClipImageEncoder",
    "fingerprint_checkpoint",
    "load_tar_images",
]
