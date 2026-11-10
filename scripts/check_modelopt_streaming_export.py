#!/usr/bin/env python3
"""Tiny real-CUDA regression probe for ModelOpt 0.47 streaming export.

The default invocation is dependency free and prints the execution plan.  The
``--execute`` path intentionally requires the pinned ModelOpt wheel, CUDA, and
Accelerate; unsupported environments return a non-success status instead of a
false positive.  Exported tensor values are never printed or retained.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import shutil
import struct
import tempfile
from pathlib import Path
from typing import Any

SCHEMA = 2
SEED = 19031
REQUIRED_CUDA_BYTES = 4 * 1024**3
AUX_NAMES = ("visual_aux", "mtp_aux")
_AUX_HASHES = {
    hashlib.sha256(f"{name}.{suffix}".encode()).hexdigest()[:12]
    for name in AUX_NAMES
    for suffix in ("weight", "weight_scale", "weight_scale_2")
}


def plan() -> dict[str, Any]:
    return {
        "schema": 1,
        "status": "planned",
        "execute_required": True,
        "model": {
            "family": "LlamaForCausalLM",
            "hidden": 32,
            "intermediate": 64,
            "layers": 1,
            "heads": 4,
            "vocab": 32,
            "aux": list(AUX_NAMES),
        },
        "quantization": "ModelOptNVFP4max",
        "arms": ["vendor_export", "streaming_export_compat"],
        "offload": "accelerate.cpu_offload(decoder_layer, execution_device=cuda)",
        "audit": "safetensors headers plus full SHA-256 file hashes; aggregate metadata only",
        "raw_tensor_output": False,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _key_hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def summarize_headers(root: Path) -> dict[str, Any]:
    """Summarize safetensors headers without exposing tensor values."""
    rows: list[dict[str, Any]] = []
    shards: list[dict[str, str]] = []
    for path in sorted(root.rglob("*.safetensors")):
        with path.open("rb") as handle:
            size_raw = handle.read(8)
            if len(size_raw) != 8:
                raise ValueError("truncated safetensors header length")
            size = struct.unpack("<Q", size_raw)[0]
            header_raw = handle.read(size)
            if len(header_raw) != size:
                raise ValueError("truncated safetensors header")
            header = json.loads(header_raw)
        shards.append({"name_hash": hashlib.sha256(path.name.encode()).hexdigest()[:16],
                       "header_sha256": hashlib.sha256(header_raw).hexdigest(),
                       "sha256": _sha256(path)})
        for key, info in header.items():
            if key == "__metadata__":
                continue
            rows.append({
                "key_hash": _key_hash(key),
                "dtype": str(info.get("dtype", "unknown")),
                "shape": list(info.get("shape", [])),
            })
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["dtype"]] = counts.get(row["dtype"], 0) + 1
    aux = [row for row in rows if row["key_hash"] in _AUX_HASHES]
    keys = [row["key_hash"] for row in aux]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate auxiliary tensor key across shards")
    return {
        "tensor_count": len(rows),
        "dtype_counts": counts,
        "aux_rows": aux,
        "shard_hashes": shards,
    }


def validate_export_report(report: dict[str, Any], *, compat: bool) -> None:
    """Fail closed on the tiny auxiliary export contract."""
    if report.get("status") != "ok":
        raise AssertionError("export status is not ok")
    aux_rows = (report.get("headers") or {}).get("aux_rows", [])
    hashes = [row["key_hash"] for row in aux_rows]
    if len(hashes) != len(set(hashes)):
        raise ValueError("duplicate auxiliary tensor key")
    rows = {row["key_hash"]: row for row in aux_rows}
    for name in AUX_NAMES:
        weight = rows.get(_key_hash(f"{name}.weight"))
        if compat:
            if weight is None or weight.get("dtype") not in {"U8", "uint8", "torch.uint8"}:
                raise AssertionError(f"{name} packed dtype contract failed")
            if weight.get("shape") != [32, 16]:
                raise AssertionError(f"{name} packed shape contract failed")
            for suffix in ("weight_scale", "weight_scale_2"):
                scale = rows.get(_key_hash(f"{name}.{suffix}"))
                if scale is None:
                    raise AssertionError(f"{name} scale contract failed")
                if suffix == "weight_scale" and scale.get("dtype") not in {
                    "F8_E4M3", "float8_e4m3fn", "torch.float8_e4m3fn"
                }:
                    raise AssertionError(f"{name} block scale dtype contract failed")
                if suffix == "weight_scale" and scale.get("shape") != [32, 2]:
                    raise AssertionError(f"{name} block scale shape contract failed")
                if suffix == "weight_scale_2" and scale.get("shape") != []:
                    raise AssertionError(f"{name} scalar scale shape contract failed")
                if suffix == "weight_scale_2" and scale.get("dtype") not in {
                    "F32", "float32", "torch.float32"
                }:
                    raise AssertionError(f"{name} scalar scale dtype contract failed")
        else:
            if weight is None or weight.get("dtype") not in {"BF16", "bfloat16", "torch.bfloat16"}:
                raise AssertionError(f"{name} vendor dtype contract failed")
            if weight.get("shape") != [32, 32]:
                raise AssertionError(f"{name} vendor shape contract failed")


def _make_model(torch: Any, transformers: Any) -> Any:
    from torch import nn
    cfg = transformers.LlamaConfig(
        vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=32,
        pad_token_id=0, bos_token_id=1, eos_token_id=2,
        architectures=["LlamaForCausalLM"], torch_dtype="bfloat16",
    )
    model = transformers.LlamaForCausalLM(cfg).to(device="cuda", dtype=torch.bfloat16)
    model.visual_aux = nn.Linear(32, 32, bias=False, device="cuda", dtype=torch.bfloat16)
    model.mtp_aux = nn.Linear(32, 32, bias=False, device="cuda", dtype=torch.bfloat16)
    return model


def _quantize(model: Any, torch: Any, mtq: Any, ids: Any, aux_input: Any) -> Any:
    import copy
    cfg = copy.deepcopy(mtq.NVFP4_DEFAULT_CFG)
    entries = cfg.setdefault("quant_cfg", [])
    if not isinstance(entries, list):
        raise TypeError("quant_cfg is not a list")
    entries.extend([{"quantizer_name": "*visual_aux*", "enable": True},
                    {"quantizer_name": "*mtp_aux*", "enable": True}])

    def loop(calibration_model: Any) -> None:
        calibration_model(input_ids=ids)
        calibration_model.visual_aux(aux_input)
        calibration_model.mtp_aux(aux_input)

    return mtq.quantize(model, cfg, forward_loop=loop)


def _assert_aux_ready(model: Any, torch: Any) -> None:
    for name in AUX_NAMES:
        module = getattr(model, name)
        weight = module.weight
        if weight.dtype is not torch.bfloat16 or weight.device.type != "cuda":
            raise AssertionError("resident auxiliary weight is not CUDA BF16")
        quantizer = getattr(module, "weight_quantizer", None)
        if quantizer is None or not bool(getattr(quantizer, "is_enabled", False)):
            raise AssertionError("auxiliary NVFP4 weight quantizer is not enabled")
        bits = getattr(quantizer, "num_bits", None)
        block_sizes = getattr(quantizer, "block_sizes", {})
        if (
            bits not in ((2, 1), [2, 1])
            or block_sizes.get(-1) != 16
            or block_sizes.get("type") != "dynamic"
            or block_sizes.get("scale_bits") != (4, 3)
        ):
            raise AssertionError("auxiliary weight quantizer is not NVFP4")


def _offload(model: Any, torch: Any) -> None:
    from accelerate import cpu_offload
    cpu_offload(model.model.layers[0], execution_device=torch.device("cuda"))


def _input_fingerprint(torch: Any, model: Any, ids: Any, aux_input: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    for tensor in (ids, aux_input):
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _export_arm(model: Any, output: Path, compat: bool, torch: Any) -> dict[str, Any]:
    from modelopt.torch.export import export_hf_checkpoint

    from megaquant.backends.modelopt import _has_accelerate_offload
    from megaquant.modelopt_streaming import streaming_export_compat
    output.mkdir(parents=True, exist_ok=True)
    if not _has_accelerate_offload(model):
        raise AssertionError("accelerate offload was not detected")
    cm = streaming_export_compat(offloaded=True) if compat else contextlib.nullcontext()
    with cm, torch.inference_mode():
        export_hf_checkpoint(model, export_dir=str(output))
    report = {"status": "ok", "offload_detected": bool(_has_accelerate_offload(model)),
              "headers": summarize_headers(output)}
    validate_export_report(report, compat=compat)
    return report


def _failure(exc: BaseException) -> dict[str, str]:
    message = str(exc).encode()
    return {"status": "error", "error_type": type(exc).__name__,
            "error_sha256": hashlib.sha256(message).hexdigest()}


def execute() -> dict[str, Any]:
    try:
        import accelerate  # noqa: F401
        import modelopt.torch.quantization as mtq
        import torch
        import transformers

        import megaquant.modelopt_streaming  # noqa: F401
        version = importlib.metadata.version("nvidia-modelopt")
    except Exception as exc:
        return {"schema": SCHEMA, "status": "unsupported", "reason": type(exc).__name__,
                "reason_sha256": hashlib.sha256(str(exc).encode()).hexdigest()}
    if version != "0.47.0":
        return {"schema": SCHEMA, "status": "unsupported", "reason": "requires_modelopt_0_47_0"}
    if not torch.cuda.is_available():
        return {"schema": SCHEMA, "status": "unsupported", "reason": "cuda_unavailable",
                "cuda_required_bytes": REQUIRED_CUDA_BYTES}
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes <= REQUIRED_CUDA_BYTES:
        return {"schema": SCHEMA, "status": "unsupported", "reason": "insufficient_cuda_memory",
                "cuda_free_bytes": free_bytes, "cuda_required_bytes": REQUIRED_CUDA_BYTES}
    tmp = Path(tempfile.mkdtemp(prefix="tiny-streaming-gpu-"))
    try:
        torch.manual_seed(SEED)
        ids = torch.randint(0, 32, (1, 8), device="cuda")
        aux_input = torch.randn(4, 32, device="cuda", dtype=torch.bfloat16)
        reports: dict[str, Any] = {}
        for name, compat in (("vendor_export", False), ("streaming_export_compat", True)):
            model = None
            try:
                torch.manual_seed(SEED)
                model = _make_model(torch, transformers)
                fingerprint = _input_fingerprint(torch, model, ids, aux_input)
                model = _quantize(model, torch, mtq, ids, aux_input)
                _assert_aux_ready(model, torch)
                _offload(model, torch)
                reports[name] = _export_arm(model, tmp / name, compat, torch)
                reports[name]["input_sha256"] = fingerprint
            except Exception as exc:
                reports[name] = _failure(exc)
            finally:
                if model is not None:
                    del model
                torch.cuda.empty_cache()
        ok = all(report.get("status") == "ok" for report in reports.values())
        fingerprints = {report.get("input_sha256") for report in reports.values()}
        input_match = len(fingerprints) == 1
        if not input_match:
            ok = False
            reports["input_match"] = {"status": "error", "error_type": "InputMismatch"}
        return {"schema": SCHEMA, "status": "executed" if ok else "failed",
                "modelopt": version, "raw_tensor_output": False, "arms": reports,
                "seed": SEED, "input_match": input_match,
                "cuda_free_bytes": free_bytes, "cuda_required_bytes": REQUIRED_CUDA_BYTES}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="run on a real CUDA device")
    args = parser.parse_args()
    result = execute() if args.execute else plan()
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    if args.execute and result.get("status") != "executed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
