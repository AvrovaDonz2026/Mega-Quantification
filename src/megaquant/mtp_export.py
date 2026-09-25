"""Keep unquantized BF16 MTP tensors in a Hugging Face export.

``quantize_mtp: false`` means leave Multi-Token Prediction in BF16, not drop
it. ModelOpt ``export_hf_checkpoint`` often omits CPU-pinned / ignored
``mtp.*`` weights. This module copies them back from the in-memory model, then
from the original Hugging Face source — only the shards whose index entries
are ``mtp.*``, never the full 27B.

SGLang's Qwen3.5 target loader skips names containing ``mtp``. The draft
loader consumes top-level ``mtp.*`` keys and treats them as BF16 unless
``quantized_layers`` lists ``mtp.*``. Mixed rewrite already excludes ``mtp*``.

SGLang 0.5.20 does not shrink Qwen3.5 hybrid depth for the draft ModelConfig,
so a sibling 1-layer draft directory is written next to the export.
"""

from __future__ import annotations

import json
import os
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from megaquant.exceptions import BackendError, MegaQuantError
from megaquant.sglang_export import load_weight_map

_SCALE_MARKERS = (
    ".weight_scale",
    ".weight_scale_2",
    ".input_scale",
    ".input_global_scale",
    ".output_scale",
    ".amax",
    "inv_freq",
)

MTP_SHARD_NAME = "mtp.safetensors"
INDEX_NAME = "model.safetensors.index.json"
NOTE_NAME = "mtp_export.json"
DRAFT_SUFFIX = "-draft"

_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)


@dataclass(frozen=True)
class RawTensor:
    dtype: str
    shape: tuple[int, ...]
    data: bytes

    @property
    def nbytes(self) -> int:
        return len(self.data)


def is_mtp_tensor_name(name: str) -> bool:
    lowered = name.replace("\\", ".")
    if "mtp." not in lowered and not lowered.startswith("mtp"):
        return False
    if "mtp." not in lowered:
        return False
    return not any(marker in lowered for marker in _SCALE_MARKERS)


def to_hf_mtp_name(name: str) -> str | None:
    """Map in-memory / sharded keys onto the public HF ``mtp.*`` names."""
    lowered = name.replace("\\", ".")
    if not is_mtp_tensor_name(lowered):
        return None
    idx = lowered.find("mtp.")
    if idx < 0:
        return None
    return lowered[idx:]


def config_mtp_layers(config: dict[str, Any] | None) -> int:
    if not isinstance(config, dict):
        return 0
    for node in (config, config.get("text_config"), config.get("language_config")):
        if not isinstance(node, dict):
            continue
        raw = node.get("mtp_num_hidden_layers")
        if raw is None:
            continue
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 0
    return 0


def weight_map_mtp_keys(weight_map: dict[str, Any]) -> list[str]:
    return sorted(key for key in weight_map if is_mtp_tensor_name(str(key)))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_safetensors(path: Path, tensors: dict[str, RawTensor]) -> None:
    """Write a safetensors file with the stdlib (no torch / safetensors pkg)."""
    if not tensors:
        raise ValueError("no tensors to write")
    header: dict[str, Any] = {}
    blobs: list[bytes] = []
    offset = 0
    for name, tensor in tensors.items():
        end = offset + tensor.nbytes
        header[name] = {
            "dtype": tensor.dtype,
            "shape": list(tensor.shape),
            "data_offsets": [offset, end],
        }
        blobs.append(tensor.data)
        offset = end
    header["__metadata__"] = {"format": "pt"}
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (8 - (len(encoded) % 8)) % 8
    if pad:
        encoded = encoded + (b" " * pad)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(blobs))


def read_safetensors(path: Path) -> dict[str, RawTensor]:
    blob = path.read_bytes()
    if len(blob) < 8:
        raise ValueError(f"{path} is not a safetensors file")
    header_len = struct.unpack_from("<Q", blob, 0)[0]
    start = 8
    end = 8 + header_len
    if end > len(blob):
        raise ValueError(f"{path} safetensors header is truncated")
    header = json.loads(blob[start:end].decode("utf-8"))
    raw = blob[end:]
    out: dict[str, RawTensor] = {}
    if not isinstance(header, dict):
        return out
    for name, info in header.items():
        if name == "__metadata__" or not isinstance(info, dict):
            continue
        offsets = info.get("data_offsets")
        if not (isinstance(offsets, list) and len(offsets) == 2):
            continue
        lo, hi = int(offsets[0]), int(offsets[1])
        shape_raw = info.get("shape") or []
        shape = tuple(int(x) for x in shape_raw)
        dtype = str(info.get("dtype") or "BF16")
        out[name] = RawTensor(dtype=dtype, shape=shape, data=raw[lo:hi])
    return out


def _torch_dtype_to_st(dtype: Any) -> str | None:
    key = str(dtype).replace("torch.", "").lower()
    return {
        "bfloat16": "BF16",
        "float16": "F16",
        "half": "F16",
        "float32": "F32",
        "float64": "F64",
    }.get(key)


def raw_tensor_from_value(value: Any) -> RawTensor | None:
    if isinstance(value, RawTensor):
        return value
    detach = getattr(value, "detach", None)
    if not callable(detach):
        return None
    tensor = detach()
    cpu = getattr(tensor, "cpu", None)
    if callable(cpu):
        tensor = cpu()
    contig = getattr(tensor, "contiguous", None)
    if callable(contig):
        tensor = contig()
    shape = tuple(int(x) for x in getattr(tensor, "shape", ()))
    st_dtype = _torch_dtype_to_st(getattr(tensor, "dtype", None))
    if st_dtype is None:
        return None
    numel = getattr(tensor, "numel", None)
    elsize = getattr(tensor, "element_size", None)
    if not callable(numel) or not callable(elsize):
        return None
    nbytes = int(numel()) * int(elsize())
    data_ptr = getattr(tensor, "data_ptr", None)
    if not callable(data_ptr):
        return None
    import ctypes

    return RawTensor(dtype=st_dtype, shape=shape, data=ctypes.string_at(data_ptr(), nbytes))


def collect_mtp_from_model(model: Any) -> dict[str, RawTensor]:
    out: dict[str, RawTensor] = {}
    named = getattr(model, "named_parameters", None)
    if not callable(named):
        return out
    for name, param in named():
        hf = to_hf_mtp_name(str(name))
        if hf is None:
            continue
        raw = raw_tensor_from_value(param)
        if raw is not None:
            out[hf] = raw
    buffers = getattr(model, "named_buffers", None)
    if callable(buffers):
        for name, buf in buffers():
            hf = to_hf_mtp_name(str(name))
            if hf is None or hf in out:
                continue
            raw = raw_tensor_from_value(buf)
            if raw is not None:
                out[hf] = raw
    return dict(sorted(out.items()))


def _index_path_for_source(source: str | Path) -> Path | None:
    root = Path(source)
    if root.is_dir() and (root / INDEX_NAME).is_file():
        return root / INDEX_NAME
    if root.is_file() and root.name == INDEX_NAME:
        return root
    return None


def _hub_download(repo_id: str, filename: str) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise BackendError(
            "huggingface_hub is required to restore MTP tensors from a Hub id"
        ) from exc
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    return Path(path)


def _source_index(source: str | Path) -> tuple[dict[str, Any], Path | str]:
    local = _index_path_for_source(source)
    if local is not None:
        return _read_json(local), local.parent
    repo = str(source).strip()
    index_file = _hub_download(repo, INDEX_NAME)
    return _read_json(index_file), repo


def _open_source_shard(root: Path | str, filename: str) -> Path:
    if isinstance(root, Path):
        path = root / filename
        if path.is_file():
            return path
        raise FileNotFoundError(path)
    return _hub_download(str(root), filename)


def collect_mtp_from_source(source: str | Path) -> dict[str, RawTensor]:
    index, root = _source_index(source)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        return {}
    shards: dict[str, list[str]] = {}
    for key, filename in weight_map.items():
        hf = to_hf_mtp_name(str(key))
        if hf is None:
            continue
        shards.setdefault(str(filename), []).append(hf)
    out: dict[str, RawTensor] = {}
    for filename, expected in shards.items():
        shard = _open_source_shard(root, filename)
        loaded = read_safetensors(shard)
        for key, tensor in loaded.items():
            hf = to_hf_mtp_name(key)
            if hf is None:
                continue
            out[hf] = tensor
        missing = [name for name in expected if name not in out]
        if missing:
            raise BackendError(
                f"MTP shard {filename} is missing {len(missing)} keys "
                f"(first {missing[0]})"
            )
    return dict(sorted(out.items()))


def update_export_index(
    export_dir: Path, shard_name: str, tensor_sizes: Mapping[str, int]
) -> None:
    index_path = export_dir / INDEX_NAME
    if index_path.is_file():
        index = _read_json(index_path)
    else:
        single = export_dir / "model.safetensors"
        weight_map = {}
        if single.is_file():
            weight_map = {key: single.name for key in read_safetensors(single)}
        index = {"weight_map": weight_map, "metadata": {}}
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        weight_map = {}
        index["weight_map"] = weight_map
    added = sum(size for name, size in tensor_sizes.items() if name not in weight_map)
    for name in tensor_sizes:
        weight_map[name] = shard_name
    metadata = index.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        index["metadata"] = metadata
    if "total_size" in metadata:
        try:
            metadata["total_size"] = int(metadata["total_size"]) + added
        except (TypeError, ValueError):
            metadata["total_size"] = added
    else:
        metadata["total_size"] = added
    _write_json(index_path, index)


def _link_or_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    try:
        os.link(src, dest)
    except OSError:
        dest.write_bytes(src.read_bytes())


def _set_if_present(node: dict[str, Any], key: str, value: Any) -> None:
    if key in node:
        node[key] = value


def _draft_text_config(config: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(json.dumps(config))
    payload.pop("quantization_config", None)
    payload.pop("compression_config", None)
    text = payload.get("text_config")
    if not isinstance(text, dict):
        text = payload
    text["num_hidden_layers"] = 1
    text["full_attention_interval"] = 1
    text["layer_types"] = ["full_attention"]
    text["mtp_num_hidden_layers"] = max(1, int(text.get("mtp_num_hidden_layers") or 1))
    _set_if_present(payload, "num_hidden_layers", 1)
    _set_if_present(payload, "full_attention_interval", 1)
    _set_if_present(payload, "layer_types", ["full_attention"])
    _set_if_present(payload, "mtp_num_hidden_layers", text["mtp_num_hidden_layers"])
    if isinstance(payload.get("text_config"), dict):
        payload["text_config"] = text
    return payload


def draft_dir_for(export_dir: Path) -> Path:
    return export_dir.parent / (export_dir.name + DRAFT_SUFFIX)


def write_mtp_draft(export_dir: str | Path, *, draft_dir: str | Path | None = None) -> Path:
    root = Path(export_dir).resolve()
    dest = Path(draft_dir).resolve() if draft_dir else draft_dir_for(root)
    weight_map = load_weight_map(root)
    mtp_keys = weight_map_mtp_keys(weight_map)
    if not mtp_keys:
        raise MegaQuantError(f"no mtp.* tensors in {root}; restore BF16 MTP first")
    shard_names = sorted({str(weight_map[key]) for key in mtp_keys})
    if len(shard_names) != 1:
        raise MegaQuantError(
            f"MTP tensors span {len(shard_names)} shards; expected a single {MTP_SHARD_NAME}"
        )
    src_shard = root / shard_names[0]
    if not src_shard.is_file():
        raise MegaQuantError(f"missing MTP shard {src_shard.name}")

    dest.mkdir(parents=True, exist_ok=True)
    config = _draft_text_config(_read_json(root / "config.json"))
    _write_json(dest / "config.json", config)
    draft_map = {key: MTP_SHARD_NAME for key in mtp_keys}
    total = 0
    if src_shard.name == MTP_SHARD_NAME:
        _link_or_copy(src_shard, dest / MTP_SHARD_NAME)
        total = src_shard.stat().st_size
    else:
        tensors = {
            key: tensor
            for key, tensor in read_safetensors(src_shard).items()
            if to_hf_mtp_name(key) in set(mtp_keys)
        }
        write_safetensors(dest / MTP_SHARD_NAME, tensors)
        total = sum(t.nbytes for t in tensors.values())
    _write_json(
        dest / INDEX_NAME,
        {"metadata": {"total_size": total}, "weight_map": draft_map},
    )
    for name in _TOKENIZER_FILES:
        src = root / name
        if src.is_file():
            _link_or_copy(src, dest / name)
    hf_quant = dest / "hf_quant_config.json"
    if hf_quant.exists():
        hf_quant.unlink()
    note = {
        "role": "sglang draft config",
        "mtp_quantized": False,
        "mtp_tensors": len(mtp_keys),
        "why": (
            "SGLang 0.5.20 does not shrink Qwen3.5 hybrid depth for the draft "
            "ModelConfig; point --speculative-draft-model-path here"
        ),
    }
    _write_json(dest / NOTE_NAME, note)
    return dest


def _public_source_id(source: str | Path | None) -> str | None:
    if source is None:
        return None
    text = str(source).strip()
    if text.count("/") == 1 and not text.startswith("/") and not text.startswith("."):
        return text
    return None


def restore_bf16_mtp(
    export_dir: str | Path,
    *,
    model: Any = None,
    source: str | Path | None = None,
    quantize_mtp: bool = False,
    write_draft: bool = True,
) -> dict[str, Any]:
    """Copy BF16 ``mtp.*`` tensors into ``export_dir`` when the recipe left MTP out.

    Idempotent if the export already has ``mtp.*`` keys. Raises if the config
    declares MTP layers but no tensors can be found.
    """
    root = Path(export_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    config = _read_json(root / "config.json")
    n_layers = config_mtp_layers(config)
    weight_map = load_weight_map(root)
    existing = weight_map_mtp_keys(weight_map)

    note: dict[str, Any] = {
        "mtp_quantized": bool(quantize_mtp),
        "mtp_num_hidden_layers": n_layers,
        "mtp_tensors": len(existing),
    }
    source_id = _public_source_id(source)
    if source_id:
        note["source_repo"] = source_id

    if quantize_mtp:
        note["method"] = "skipped-quantize-mtp"
        _write_json(root / NOTE_NAME, note)
        return note

    if existing:
        note["method"] = "already-present"
        if write_draft:
            dest = write_mtp_draft(root)
            note["draft_dir"] = dest.name
        _write_json(root / NOTE_NAME, note)
        return note

    if n_layers <= 0 and model is None and source is None:
        note["method"] = "skipped-no-mtp"
        return note

    tensors = collect_mtp_from_model(model) if model is not None else {}
    method = "in-memory"
    if not tensors and source is not None:
        tensors = collect_mtp_from_source(source)
        method = "hf-source"

    if not tensors:
        if n_layers <= 0:
            note["method"] = "skipped-no-mtp"
            return note
        raise BackendError(
            f"{root.name} config has mtp_num_hidden_layers={n_layers} but the "
            "export has no mtp.* tensors. Pass the in-memory model or the "
            "original HF source (only the MTP shard is read, not the full 27B)."
        )

    shard = root / MTP_SHARD_NAME
    write_safetensors(shard, tensors)
    update_export_index(root, MTP_SHARD_NAME, {name: t.nbytes for name, t in tensors.items()})
    note["method"] = method
    note["mtp_tensors"] = len(tensors)
    note["mtp_dtype"] = next(iter(tensors.values())).dtype
    note["mtp_tensor_bytes"] = sum(t.nbytes for t in tensors.values())
    note["shard"] = MTP_SHARD_NAME
    if write_draft:
        dest = write_mtp_draft(root)
        note["draft_dir"] = dest.name
    _write_json(root / NOTE_NAME, note)
    return note


def restore_bf16_mtp_from_recipe(
    export_dir: str | Path,
    recipe: Any,
    *,
    model: Any = None,
    write_draft: bool = True,
) -> dict[str, Any] | None:
    """Honor ``model.quantize_mtp``. No-op when the recipe opted into MTP quant."""
    from megaquant.models.base import recipe_flag, recipe_get

    if recipe_flag(recipe, "model", "quantize_mtp", default=False):
        return restore_bf16_mtp(
            export_dir,
            model=model,
            source=recipe_get(recipe, "model", "source", default=None),
            quantize_mtp=True,
            write_draft=False,
        )
    source = recipe_get(recipe, "model", "source", default=None)
    return restore_bf16_mtp(
        export_dir,
        model=model,
        source=source,
        quantize_mtp=False,
        write_draft=write_draft,
    )
