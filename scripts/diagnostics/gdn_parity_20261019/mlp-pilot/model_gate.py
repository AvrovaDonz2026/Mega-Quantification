#!/usr/bin/env python3
"""Strict stdlib gate for MSE ModelOpt MLP precision recovery exports."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path
from typing import Any

EXPECTED_PROVENANCE_SHA256 = (
    "66edb2373a176d330c70a8445c52e33b9caa0c8b05378a9cafd57a97ce19b55c"
)
_CHUNK = 8 * 1024 * 1024


def _load_tool(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("precision_ablation_gate_tool", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load precision ablation tool")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("JSON object required")
    return value


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _stream(tool: Any, parts: tuple[Any, ...]) -> str:
    digest = hashlib.sha256()
    for tensor in parts:
        with tensor.path.open("rb") as handle:
            handle.seek(tensor.offset)
            remaining = tensor.nbytes
            while remaining:
                block = handle.read(min(_CHUNK, remaining))
                if not block:
                    raise RuntimeError("truncated tensor payload")
                digest.update(block)
                remaining -= len(block)
    return digest.hexdigest()


def _inventory(tool: Any, tensors: dict[str, Any]) -> str:
    return hashlib.sha256(
        "".join(f"{name} {_stream(tool, (tensors[name],))}\n" for name in sorted(tensors)).encode()
    ).hexdigest()


def _tokenizer_files(root: Path) -> set[str]:
    names = set()
    for path in root.iterdir():
        if not path.is_file():
            continue
        name = path.name.lower()
        if (
            name.startswith("tokenizer")
            or name in {"special_tokens_map.json", "added_tokens.json"}
            or name in {"spiece.model", "sentencepiece.bpe.model"}
            or name in {"vocab.json", "merges.txt"}
        ):
            names.add(path.name)
    return names


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_model(
    export: Path,
    bf16: Path,
    model: Path,
    variant: str,
    ablation_tool: Path,
) -> dict[str, Any]:
    """Verify a derived control/MLP export and return scalar proof fields only."""
    if variant not in {"control", "mlp"}:
        raise RuntimeError("variant must be control or mlp")
    export, bf16, model, ablation_tool = (
        Path(export).resolve(), Path(bf16).resolve(), Path(model).resolve(), Path(ablation_tool).resolve()
    )
    tool = _load_tool(ablation_tool)
    plan = tool.prepare_plan(export, bf16, variant)
    original = plan["original"]
    expected = plan["result"]
    actual, actual_index = tool.read_checkpoint(model)
    _assert(set(actual) == set(expected), "derived tensor inventory differs from plan")
    expected_hashes: dict[str, str] = {}
    for name, tensor in expected.items():
        source_parts = plan["replacements"].get(name)
        expected_hashes[name] = _stream(tool, source_parts if source_parts else (original[name],))
        got = actual[name]
        _assert(got.dtype == tensor.dtype and got.shape == tensor.shape, "tensor metadata mismatch")
        _assert(_stream(tool, (got,)) == expected_hashes[name], "tensor payload hash mismatch")
    _assert(not set(actual).intersection(plan["removed"]), "removed scale tensor retained")

    config = _json(model / "config.json")
    hf = _json(model / "hf_quant_config.json")
    hf_quant = hf.get("quantization", hf)
    config_quant = config.get("quantization_config")
    _assert(isinstance(config_quant, dict) and isinstance(hf_quant, dict), "invalid quant metadata")
    _assert(hf_quant.get("quant_algo") == "MIXED_PRECISION", "HF quant algorithm mismatch")
    _assert(config_quant.get("quant_algo") == "MIXED_PRECISION", "config quant algorithm mismatch")
    layers = plan["layers"]
    _assert(hf_quant.get("quantized_layers") == layers, "HF quantized_layers mismatch")
    _assert(config_quant.get("quantized_layers") == layers, "config quantized_layers mismatch")
    with tempfile.TemporaryDirectory(prefix="mlp-gate-") as staging:
        tool._metadata(export, Path(staging), layers)
        _assert(_json(model / "config.json") == _json(Path(staging) / "config.json"), "config transform mismatch")
        _assert(_json(model / "hf_quant_config.json") == _json(Path(staging) / "hf_quant_config.json"), "HF config transform mismatch")
    selected_modules = [name for name in layers if tool.selected(name, variant)]
    _assert(len(selected_modules) == 0, "selected layers remain quantized")

    visual = [name for name in layers if name.startswith(("visual.", "model.visual."))]
    mtp = [name for name in layers if name.startswith("mtp.")]
    _assert(len(visual) == 110 and len(mtp) == 8, "auxiliary quantized projection count mismatch")
    expected_restored = 0 if variant == "control" else 192
    _assert(plan["summary"].get("restored_modules") == expected_restored, "restored module count mismatch")

    provenance_name = "provenance.json"
    original_provenance = export / provenance_name
    model_provenance = model / provenance_name
    _assert(original_provenance.is_file() and model_provenance.is_file(), "provenance missing")
    _assert(_sha(original_provenance) == EXPECTED_PROVENANCE_SHA256, "original provenance SHA mismatch")
    _assert(_sha(model_provenance) == _sha(original_provenance), "provenance bytes changed")
    provenance = _json(original_provenance)
    _assert(
        provenance.get("algorithm") == "mse"
        and provenance.get("backend") == "modelopt"
        and provenance.get("scheme") == "nvfp4_w4a4"
        and provenance.get("calibration", {}).get("with_images") is True,
        "original provenance recipe mismatch",
    )

    for name in _tokenizer_files(export) | _tokenizer_files(model):
        _assert((export / name).is_file() and (model / name).is_file(), "tokenizer ancillary inventory mismatch")
        _assert(_sha(export / name) == _sha(model / name), "tokenizer ancillary bytes changed")

    record = _json(model / "precision_ablation.json")
    summary = plan["summary"]
    _assert(record.get("variant") == variant, "ablation variant mismatch")
    for key in ("restored_modules", "restored_tensors", "removed_scale_tensors", "preserved_tensors", "output_tensor_bytes"):
        _assert(record.get(key) == summary.get(key), "ablation statistic mismatch")
    _assert(record.get("payloads_verified") is True, "ablation payload verification missing")
    _assert(record.get("tensor_inventory_sha256") == _inventory(tool, actual), "output inventory mismatch")
    _assert(record.get("input_tensor_inventory_sha256") == _inventory(tool, original), "input inventory mismatch")
    _assert(record.get("input_metadata_sha256", {}).get("config.json") == _sha(export / "config.json"), "input config inventory mismatch")
    _assert(record.get("input_metadata_sha256", {}).get("hf_quant_config.json") == _sha(export / "hf_quant_config.json"), "input HF config inventory mismatch")
    _assert(record.get("input_metadata_sha256", {}).get("model.safetensors.index.json") == _sha(export / "model.safetensors.index.json"), "input index inventory mismatch")
    expected_map = {name: filename for name, filename in plan["index"]["weight_map"].items() if name in expected}
    _assert(actual_index.get("weight_map") == expected_map, "derived weight map mismatch")
    _assert(record.get("quantized_layer_counts") == summary.get("quantized_layer_counts"), "quantized layer statistics mismatch")
    _assert(record.get("requires_quantization") == "modelopt_mixed", "quantization requirement mismatch")
    return {
        "passed": True,
        "variant_mlp": variant == "mlp",
        "restored_modules": int(summary["restored_modules"]),
        "tensor_count": len(actual),
        "visual_quantized_layers": len(visual),
        "mtp_quantized_layers": len(mtp),
        "provenance_sha256_ok": True,
        "provenance_sha256": _sha(original_provenance),
        "input_inventory_sha256": record["input_tensor_inventory_sha256"],
        "output_inventory_sha256": record["tensor_inventory_sha256"],
        "config_sha256": _sha(model / "config.json"),
        "hf_quant_config_sha256": _sha(model / "hf_quant_config.json"),
        "payload_sha256_count": len(actual),
        "tokenizer_files": len(_tokenizer_files(model)),
    }

