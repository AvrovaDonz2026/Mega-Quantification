#!/usr/bin/env python3
"""Cold-cache natural-start MSE/MLP sensitivity pilot; persist only summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import socket
import sys
import urllib.request
from pathlib import Path
from types import SimpleNamespace

IMAGE = "sha256:e98ca3c6444896d8800dc63cad3d4d22ee842af32fb16badd6cc04ee9090babf"
REVISION = "49d3acfd4d54b0cf1afffcfeb369c63af4f56499"
NATIVE_SHA = "8e151d76fdd25603216aeb79c59a39d637ad25c47efd87a3581f6b65b3dc7218"
PROMPT_SHA = "20e75efb3ff7b137eee5484d35b30e0bb3668d8cc810b36990c7d3cae662d436"
KNOWN_PERIOD = 2032
RANDOM_SEED = 20261020


def _strict_ids(value, source):
    """Normalize an EOS id or list, rejecting malformed metadata."""
    values = value if isinstance(value, list) else [value]
    if value is None:
        return []
    if any(type(item) is not int or item < 0 for item in values):
        raise RuntimeError(f"malformed {source} EOS metadata")
    return values


def _metadata_sha(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_eos_metadata(model):
    """Load EOS ids and privacy-preserving provenance from a model directory."""
    config = json.loads((model / "config.json").read_text())
    generation_path = model / "generation_config.json"
    tokenizer_config_path = model / "tokenizer_config.json"
    tokenizer_path = model / "tokenizer.json"
    sources = []
    eos = set()

    if generation_path.exists():
        generation = json.loads(generation_path.read_text())
        ids = _strict_ids(generation.get("eos_token_id"), "generation_config")
        eos.update(ids)
        sources.append(
            {
                "source": "generation_config",
                "count": len(ids),
                "sha256": _metadata_sha(ids),
                "present": True,
            }
        )
    else:
        sources.append(
            {
                "source": "generation_config",
                "count": 0,
                "sha256": _metadata_sha([]),
                "present": False,
            }
        )
    config_eos = config.get("eos_token_id")
    if config_eos is None:
        config_eos = config.get("text_config", {}).get("eos_token_id")
    ids = _strict_ids(config_eos, "config")
    eos.update(ids)
    sources.append(
        {"source": "config", "count": len(ids), "sha256": _metadata_sha(ids), "present": bool(ids)}
    )

    tokenizer_ids = []
    additional_ids = []
    additional_tokens = {"<|eom_id|>", "<|content_model_end_sampling|>", "<|ifm|im_end|>"}
    tokenizer_token = None
    tokenizer_config = {}
    if tokenizer_config_path.exists():
        tokenizer_config = json.loads(tokenizer_config_path.read_text())
        tokenizer_token = tokenizer_config.get("eos_token")
        if isinstance(tokenizer_token, dict):
            tokenizer_token = tokenizer_token.get("content")
        if tokenizer_token is not None and not isinstance(tokenizer_token, str):
            raise RuntimeError("malformed tokenizer EOS metadata")
    if tokenizer_path.exists():
        tokenizer = json.loads(tokenizer_path.read_text())
        for item in tokenizer.get("added_tokens", []):
            if (
                tokenizer_token is not None
                and isinstance(item, dict)
                and item.get("content") == tokenizer_token
            ):
                tokenizer_ids.extend(_strict_ids(item.get("id"), "tokenizer"))
            if isinstance(item, dict) and item.get("content") in additional_tokens:
                additional_ids.extend(_strict_ids(item.get("id"), "tokenizer additional stop"))
    eos.update(tokenizer_ids)
    sources.append(
        {
            "source": "tokenizer",
            "count": len(tokenizer_ids),
            "sha256": _metadata_sha(tokenizer_ids),
            "present": bool(tokenizer_ids),
        }
    )
    eos.update(additional_ids)
    sources.append(
        {
            "source": "tokenizer_additional_stop",
            "count": len(additional_ids),
            "sha256": _metadata_sha(additional_ids),
            "present": bool(additional_ids),
        }
    )
    if not eos:
        raise RuntimeError("termination token metadata missing")
    return eos, {"sources": sources, "any_present": True}


def lcp(left, right):
    return next(
        (i for i, (x, y) in enumerate(zip(left, right, strict=False)) if x != y),
        min(len(left), len(right)),
    )


def known_period_stats(tokens, lag=KNOWN_PERIOD):
    """Return scalar statistics for the longest contiguous lag-equality run.

    ``count`` is the number of consecutive comparisons satisfying
    ``tokens[i] == tokens[i - lag]``.  The corresponding extent includes the
    first lag-sized period, so ``extent = count + lag`` when a run exists.
    This detects only the supplied lag; it does not search for other periods.
    """
    if lag <= 0:
        raise ValueError("period lag must be positive")
    best_count = 0
    best_onset = None
    current_count = 0
    current_onset = None
    for index in range(lag, len(tokens)):
        if tokens[index] == tokens[index - lag]:
            if current_count == 0:
                current_onset = index - lag
            current_count += 1
            if current_count > best_count:
                best_count = current_count
                best_onset = current_onset
        else:
            current_count = 0
            current_onset = None
    extent = best_count + lag if best_count else 0
    complete_periods = extent // lag if extent else 0
    return {
        "periodic_lag": lag,
        "periodic_onset": best_onset,
        "periodic_count": best_count,
        "periodic_extent": extent,
        "periodic_complete_periods": complete_periods,
        "periodic_confirmed": complete_periods >= 2,
    }


def termination_stats(out, meta, eos, think_end, budget):
    reason = meta.get("finish_reason")
    if not isinstance(reason, dict) or reason.get("type") not in {"stop", "length"}:
        raise RuntimeError("invalid finish reason")
    if reason["type"] == "length" and (
        len(out) != budget or ("length" in reason and reason["length"] != budget)
    ):
        raise RuntimeError("length finish does not match output budget")
    terminal_eos = bool(out and out[-1] in eos)
    first_end = next((index for index, token in enumerate(out) if token in think_end), None)
    matched = reason.get("matched")
    matched_kind = (
        "id"
        if type(matched) is int
        else "text"
        if isinstance(matched, str)
        else "none"
        if matched is None
        else "unknown"
    )
    matched_is_eos = type(matched) is int and matched in eos
    matched_equals_terminal = type(matched) is int and bool(out) and matched == out[-1]
    return {
        "finish_reason_type": reason["type"],
        "contains_eos": any(token in eos for token in out),
        "contains_thinking_end": first_end is not None,
        "thinking_end_count": sum(token in think_end for token in out),
        "terminal_eos": terminal_eos,
        "first_thinking_end_index": first_end,
        "natural_termination": (
            reason["type"] == "stop"
            and terminal_eos
            and first_end is not None
            and first_end < len(out) - 1
            and (matched is None or matched_equals_terminal)
        ),
        "finish_reason_matched": {
            "match_kind": matched_kind,
            "matched_is_eos": matched_is_eos,
            "matched_equals_terminal": matched_equals_terminal,
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--remote-root", type=Path, required=True)
    p.add_argument("--export", type=Path, required=True)
    p.add_argument("--bf16", type=Path, required=True)
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--mlp", type=Path, required=True)
    p.add_argument("--ablation-tool", type=Path, required=True)
    p.add_argument("--native-private", type=Path, required=True)
    p.add_argument("--phase", choices=("early", "long"), default="early")
    p.add_argument("--budget", type=int, default=None)
    o = p.parse_args()
    if o.phase == "early":
        o.budget = 4096 if o.budget is None else o.budget
        if not 512 <= o.budget <= 8192:
            p.error("early pilot budget must be between 512 and 8192")
    else:
        o.budget = 49152 if o.budget is None else o.budget
        if o.budget != 49152:
            p.error("long phase requires --budget 49152")
    generate_timeout = 3600 if o.phase == "early" else 14400
    o.root.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(o.remote_root / "candidate-bundle-r1/parity-v7"))
    import run_v7 as v
    from model_gate import verify_model

    def progress(state, **fields):
        v.write_json(o.root / "status.json", {"state": state, "at_utc": v.now(), **fields})

    def get(port, path):
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=30) as r:
            return json.load(r)

    progress("preflight")
    try:
        if v.sha256_file(o.native_private) != NATIVE_SHA:
            raise RuntimeError("native fixture SHA differs")
        native = json.loads(o.native_private.read_text())
        if len(native["cases"]) != 1 or native["cases"][0]["case"] != 3:
            raise RuntimeError("anonymous fixture identity differs")
        old = native["cases"][0]
        ids = old["request"]["input_ids"]
        reference = old["response"]["output_ids"]
        if len(ids) != 288 or v.digest(ids) != PROMPT_SHA or len(reference) != 49152:
            raise RuntimeError("native prompt/output identity differs")
        if any(type(x) is not int or x < 0 for x in ids + reference):
            raise RuntimeError("native token type invalid")
        report = {
            "schema_version": 1,
            "started_at_utc": v.now(),
            "case": "anonymous-case3",
            "phase": o.phase,
            "controller_sha256": v.sha256_file(Path(__file__)),
            "scope": (
                "Natural-start early-trajectory pilot. Finite budget does not establish "
                "long-loop repair, GPQA accuracy, or stopping recovery."
                if o.phase == "early"
                else "Natural-start long-budget trajectory from the original prompt. "
                "Known-period scalars detect only lag-2032 repetition; stopping does "
                "not establish answer correctness, and neither a detected loop nor "
                "the absence of one within this budget establishes repair."
            ),
            "native_private_sha256": NATIVE_SHA,
            "prompt_ids_sha256": PROMPT_SHA,
            "prompt_tokens": len(ids),
            "native_prefix_tokens": 0,
            "historical_reference_window_sha256": v.digest(reference[: o.budget]),
            "max_new_tokens": o.budget,
            "repetitions_per_arm": 2,
            "image_id": IMAGE,
            "source_revision": REVISION,
            "ablation_tool_sha256": v.sha256_file(o.ablation_tool),
            "model_gate_sha256": v.sha256_file(Path(__file__).with_name("model_gate.py")),
            "protocol": {
                "context_length": 65536,
                "temperature": 0,
                "tp_size": 1,
                "mtp": False,
                "quantization_loader": "modelopt_mixed",
                "kv_cache_dtype": "fp8_e4m3",
                "mamba_ssm_dtype": "float32",
                "disable_radix_cache": True,
                "allow_auto_truncate": False,
                "cuda_graphs": False,
                "random_seed": RANDOM_SEED,
                "FLA_USE_FAST_OPS": "0",
                "force_shared_decode": "0",
            },
            "arms": {},
        }
        outputs = {}
        for variant, model in [("control", o.control), ("mlp", o.mlp)]:
            arm_root = o.root / variant
            arm_root.mkdir()
            progress("model_gate", arm=variant)
            proof = verify_model(o.export, o.bf16, model, variant, o.ablation_tool)
            # This is a diagnostic mixed-loader model, never a W4A8 scheme label.
            scheme_label = "mse-" + variant
            a = SimpleNamespace(
                root=arm_root,
                job_id="mlp-pilot-" + variant,
                model_scheme=scheme_label,
                source_variant="all-beta",
                source_revision=REVISION,
                source=o.remote_root / "source-package/sglang-all-beta",
                source_archive=o.remote_root / "source-package/sglang-all-beta.tar.gz",
                source_manifest=o.remote_root / "source-package/sglang-all-beta.manifest.json",
                model=model,
                image=IMAGE,
                image_id=IMAGE,
                fla_use_fast_ops="0",
                force_shared_decode="0",
                health_timeout=1200,
            )
            source = v.source_gate(a)
            source.pop("source_file_hashes", None)
            eos, eos_proof = load_eos_metadata(model)
            tok = json.loads((model / "tokenizer.json").read_text())
            think_end = [
                x["id"] for x in tok.get("added_tokens", []) if x.get("content") == "</think>"
            ]
            if not eos or not think_end:
                raise RuntimeError("termination token metadata missing")
            port = 32262
            with socket.socket() as s:
                if s.connect_ex(("127.0.0.1", port)) == 0:
                    raise RuntimeError("experiment port occupied")
            if (
                v.run(["docker", "image", "inspect", "--format", "{{.Id}}", IMAGE]).stdout.strip()
                != IMAGE
            ):
                raise RuntimeError("image identity differs")
            v.gpu_idle()
            command = v.server_command(a, a.job_id, "baseline", port)
            command[command.index("--context-length") + 1] = "65536"
            command += ["--disable-radix-cache", "--random-seed", str(RANDOM_SEED)]
            server = None
            arm = {
                "model_proof": proof,
                "source_proof": source,
                "termination_metadata": eos_proof,
                "trials": [],
            }
            report["arms"][variant] = arm
            outputs[variant] = []
            try:
                progress("starting", arm=variant)
                server = v.run(command).stdout.strip()
                v.owned(server, a.job_id, "baseline", IMAGE, REVISION)
                info = v.wait_health(a, server, "baseline", port)
                model_info = get(port, "/model_info")
                expected = {
                    "host": "127.0.0.1",
                    "port": port,
                    "model_path": "/models/Qwen3.8-27B-NVFP4-" + scheme_label.upper(),
                    "quantization": "modelopt_mixed",
                    "fp4_gemm_runner_backend": "flashinfer_cutlass",
                    "attention_backend": "flashinfer",
                    "sampling_backend": "pytorch",
                    "linear_attn_backend": "triton",
                    "mamba_backend": "triton",
                    "mamba_ssm_dtype": "float32",
                    "kv_cache_dtype": "fp8_e4m3",
                    "context_length": 65536,
                    "tp_size": 1,
                    "max_running_requests": 1,
                    "enable_multimodal": True,
                    "speculative_algorithm": None,
                    "disable_radix_cache": True,
                    "allow_auto_truncate": False,
                    "random_seed": RANDOM_SEED,
                }
                if any(info.get(k) != value for k, value in expected.items()):
                    raise RuntimeError(
                        "server identity mismatch: "
                        + ",".join(k for k, val in expected.items() if info.get(k) != val)
                    )
                if model_info.get("model_path") != expected["model_path"]:
                    raise RuntimeError("model path differs")
                if any(
                    info.get(k)
                    for k in ["dump_requests_folder", "crash_dump_folder", "log_requests"]
                ):
                    raise RuntimeError("server output logging enabled")
                if any(
                    info.get("cuda_graph_config", {}).get(phase, {}).get("backend") != "disabled"
                    for phase in ["prefill", "decode"]
                ):
                    raise RuntimeError("CUDA graph enabled")
                arm["server_identity"] = {
                    "passed": True,
                    "expected": expected,
                    "server_info_sha256": v.digest(info),
                    "model_info_sha256": v.digest(model_info),
                }
                runtime = v.runtime_source_hashes(a, server)
                v.write_json(arm_root / "runtime-source.json", runtime)
                if not runtime["passed"]:
                    raise RuntimeError("runtime source hash differs")
                arm["runtime_source_sha256"] = v.sha256_file(arm_root / "runtime-source.json")
                for trial in (1, 2):
                    progress("request", arm=variant, trial=trial)
                    req = urllib.request.Request(
                        f"http://127.0.0.1:{port}/flush_cache?timeout=180",
                        data=b"{}",
                        method="POST",
                        headers={"Content-Type": "application/json"},
                    )
                    with urllib.request.urlopen(req, timeout=200) as r:
                        if r.status != 200 or not r.read().startswith(b"Cache flushed."):
                            raise RuntimeError("flush acknowledgement differs")
                    payload = {
                        "input_ids": ids,
                        "sampling_params": {"temperature": 0.0, "max_new_tokens": o.budget},
                        "stream": False,
                        "return_logprob": True,
                        "logprob_start_len": -1,
                        "return_text_in_logprobs": False,
                        "return_prompt_token_ids": True,
                        "log_metrics": False,
                    }
                    req = urllib.request.Request(
                        f"http://127.0.0.1:{port}/generate",
                        data=json.dumps(payload).encode(),
                        method="POST",
                        headers={"Content-Type": "application/json"},
                    )
                    with urllib.request.urlopen(req, timeout=generate_timeout) as r:
                        response = json.load(r)
                    out = response["output_ids"]
                    meta = response["meta_info"]
                    if (
                        not out
                        or any(type(x) is not int or x < 0 for x in out)
                        or len(out) != meta.get("completion_tokens")
                        or len(out) > o.budget
                    ):
                        raise RuntimeError("response token counts invalid")
                    if (
                        response.get("prompt_token_ids") != ids
                        or meta.get("prompt_tokens") != len(ids)
                        or meta.get("cached_tokens") != 0
                    ):
                        raise RuntimeError("prompt/cold-cache gate failed")
                    if any(
                        meta.get(key, 0) not in (0, None)
                        for key in (
                            "spec_verify_ct",
                            "num_retractions",
                            "retraction_count",
                        )
                    ):
                        raise RuntimeError("unexpected MTP or retraction")
                    lp = [x[0] for x in meta.get("output_token_logprobs", [])]
                    if len(lp) != len(out) or any(
                        type(x) not in (int, float) or not math.isfinite(x) for x in lp
                    ):
                        raise RuntimeError("output logprobs missing/nonfinite")
                    row = {
                        "trial": trial,
                        "output_tokens": len(out),
                        "output_ids_sha256": v.digest(out),
                        "prompt_echo_exact": True,
                        "cached_tokens": 0,
                        "historical_natural_lcp": lcp(out, reference[: o.budget]),
                        "historical_natural_window_exact": out == reference[: o.budget],
                        "logprobs_finite": True,
                        "mean_output_logprob": sum(lp) / len(lp),
                        "e2e_latency": meta.get("e2e_latency"),
                    }
                    row.update(termination_stats(out, meta, eos, think_end, o.budget))
                    row.update(known_period_stats(out))
                    outputs[variant].append(out)
                    arm["trials"].append(row)
                    v.write_json(arm_root / "checkpoint.json", arm)
                    del response, payload, lp
                arm["repeat_stable"] = outputs[variant][0] == outputs[variant][1]
                arm["completed_at_utc"] = v.now()
            finally:
                arm["cleanup"] = v.cleanup(server, a.job_id, "baseline", IMAGE, REVISION)
                v.write_json(arm_root / "checkpoint.json", arm)
        left, right = outputs["control"][0], outputs["mlp"][0]
        report["comparison"] = {
            "control_vs_mlp_lcp": lcp(left, right),
            "output_exact": left == right,
            "both_repeat_stable": all(x["repeat_stable"] for x in report["arms"].values()),
        }
        report["stopping_outcome"] = {
            variant: {
                "natural_termination_trials": sum(
                    trial["natural_termination"] for trial in arm["trials"]
                ),
                "all_trials_naturally_terminated": all(
                    trial["natural_termination"] for trial in arm["trials"]
                ),
            }
            for variant, arm in report["arms"].items()
        }
        report["known_cycle_outcome"] = {
            variant: {
                "confirmed_periodic_trials": sum(
                    trial["periodic_confirmed"] for trial in arm["trials"]
                ),
                "any_confirmed_periodic": any(
                    trial["periodic_confirmed"] for trial in arm["trials"]
                ),
            }
            for variant, arm in report["arms"].items()
        }
        report["status"] = "completed" if report["comparison"]["both_repeat_stable"] else "unstable"
        report["completed_at_utc"] = v.now()
        v.write_json(o.root / "report.json", report)
        progress(report["status"])
    except Exception as exc:
        progress("failed", error_type=type(exc).__name__)
        raise


if __name__ == "__main__":
    main()
