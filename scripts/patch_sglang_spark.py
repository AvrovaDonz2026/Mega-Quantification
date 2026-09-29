#!/usr/bin/env python3
"""Patch pinned SGLang 0.5.20 to load MegaQuant's quantized vision/MTP.

The upstream Qwen3.5 loader assumes BF16 vision and a BF16 MTP fc.  This
build-time patch enables them only when the exported quantized_layers map
explicitly names those modules.  Source hashes, version checks and a patch
manifest make unexpected upstream changes fail before any file is written.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
from typing import Any

VERSION = "0.5.20"
PATCH_ID = "megaquant-spark-v1"
MANIFEST = ".megaquant-spark-patch.json"
MARKER = f"# {PATCH_ID}: SGLang {VERSION} quantized vision and MTP compatibility.\n"

# Exact files from https://github.com/sgl-project/sglang/tree/v0.5.20/python/sglang
UPSTREAM_SHA256 = {
    "srt/models/qwen3_vl.py": "a32ac0793e0526c8951f2ad3936bd8b002b7ad00a291c442ad2f215b9064a80f",
    "srt/models/qwen3_5_mtp.py": "1ebc9643333046d5621ad8b1bd6bd005686988c0104576bd1ea6c3d5e51acf24",
    "srt/layers/quantization/modelopt_quant.py": (
        "afe1315c27e37dac2ecad87fe867217081609d7be4b43f0cf5eb42401d6d5d8e"
    ),
    "srt/configs/model_config.py": (
        "85c36b134e3e4a39bce382d9172aa4fe4819f02996d55cbdf779846da4cfb8c8"
    ),
}

VISION_HELPER = '''
def _megaquant_vision_quant_config(quant_config):
    """Keep BF16 vision unless the checkpoint explicitly quantized it."""
    if quant_config is None:
        return None
    name = quant_config.get_name()
    if name not in {"modelopt_mixed", "modelopt_fp4"}:
        return None
    layers = getattr(quant_config, "quantized_layers", None)
    if layers is None:
        layers = getattr(quant_config, "_megaquant_quantized_layers", {})
    if name == "modelopt_fp4" and not layers:
        # Uniform serialized NVFP4 has no layer map.  The ModelOpt config's
        # exclusion matcher is the source of truth for Spark's opt-in blocks.
        is_excluded = getattr(quant_config, "is_layer_excluded", None)
        if not callable(is_excluded) or is_excluded("visual.blocks.0.attn.qkv_proj"):
            return None
        return quant_config
    if not any(key.startswith(("model.visual.", "visual.")) for key in layers):
        return None
    if name == "modelopt_mixed":
        # The text loader has a different prefix map.  Keep its config intact.
        from copy import deepcopy

        quant_config = deepcopy(quant_config)
        for key, value in list(quant_config.quantized_layers.items()):
            if key.startswith(("model.visual.", "visual.")):
                mapped = (
                    key[: -len(".attn.qkv")] + ".attn.qkv_proj"
                    if key.endswith(".attn.qkv")
                    else key
                )
                if mapped != key:
                    quant_config.quantized_layers[mapped] = value
    return quant_config

'''

MTP_GATE_OLD = '''    if quant_config and quant_config.get_name() == "modelopt_mixed":
        # MIXED_PRECISION lists mtp.* layers only when the MTP head is quantized.
        if any(name.startswith("mtp.") for name in quant_config.quantized_layers):
            return quant_config
        return None
    if quant_config and (
        quant_config.get_name() == "modelopt_fp4"
        and quant_config.is_checkpoint_nvfp4_serialized
    ):
        return None
'''

MTP_GATE_NEW = '''    if quant_config and (
        quant_config.get_name() == "modelopt_mixed"
        or (
            quant_config.get_name() == "modelopt_fp4"
            and quant_config.is_checkpoint_nvfp4_serialized
        )
    ):
        # Serialized MTP is quantized only with explicit checkpoint evidence.
        layers = getattr(quant_config, "quantized_layers", None)
        if layers is None:
            layers = getattr(quant_config, "_megaquant_quantized_layers", {})
        if any(name.startswith("mtp.") for name in layers):
            return quant_config
        if quant_config.get_name() == "modelopt_fp4" and not layers:
            is_excluded = getattr(quant_config, "is_layer_excluded", None)
            if callable(is_excluded) and not is_excluded("mtp.fc"):
                return quant_config
        return None
'''


class PatchError(RuntimeError):
    """Installed SGLang does not match the tested patch target."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise PatchError(f"{label}: expected one upstream fragment, found {count}")
    return source.replace(old, new, 1)


def transform(relative_path: str, source: str) -> str:
    """Apply exact replacements; callers verify the complete source hash first."""
    if MARKER in source:
        raise PatchError(f"{relative_path}: already patched without a valid manifest")
    if relative_path == "srt/models/qwen3_vl.py":
        source = _replace_once(
            source,
            'class Qwen3VLForConditionalGeneration(nn.Module):\n',
            VISION_HELPER + 'class Qwen3VLForConditionalGeneration(nn.Module):\n',
            "vision helper",
        )
        source = _replace_once(
            source,
            '''                quant_config=None,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                prefix=add_prefix("model.visual", prefix),''',
            '''                quant_config=_megaquant_vision_quant_config(quant_config),
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                prefix=add_prefix("model.visual", prefix),''',
            "vision constructor",
        )
    elif relative_path == "srt/models/qwen3_5_mtp.py":
        source = _replace_once(source, MTP_GATE_OLD, MTP_GATE_NEW, "serialized MTP gate")
        source = _replace_once(
            source,
            "from sglang.srt.layers.logits_processor import LogitsProcessor\n",
            "from sglang.srt.layers.logits_processor import LogitsProcessor\n"
            "from sglang.srt.layers.linear import ReplicatedLinear\n",
            "MTP Linear import",
        )
        source = _replace_once(
            source,
            "class Qwen3_5ForCausalLMMTP(nn.Module):\n",
            "class Qwen3_5ForCausalLMMTP(nn.Module):\n"
            "    packed_modules_mapping = Qwen3_5ForCausalLM.packed_modules_mapping.copy()\n",
            "MTP fused projection map",
        )
        source = _replace_once(
            source,
            "        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)\n",
            '''        self.fc = ReplicatedLinear(
            2 * config.hidden_size, config.hidden_size, bias=False,
            quant_config=quant_config, prefix=add_prefix("mtp.fc", prefix),
        )
''',
            "MTP fc constructor",
        )
        source = _replace_once(
            source,
            "            hidden_states = self.fc(hidden_states)\n",
            "            hidden_states, _ = self.fc(hidden_states)\n",
            "MTP fc forward",
        )
        source = _replace_once(
            source,
            "        mtp_config.full_attention_interval = 1\n",
            "        mtp_config.full_attention_interval = 1\n"
            '        mtp_config.layer_types = ["full_attention"]\n',
            "MTP decoder layer type",
        )
    elif relative_path == "srt/layers/quantization/modelopt_quant.py":
        source = _replace_once(
            source,
            '        quant_config.is_w4a16 = quant_method == "W4A16_NVFP4"\n',
            '''        quant_config.is_w4a16 = quant_method == "W4A16_NVFP4"
        # Preserve explicit vision/MTP evidence from uniform NVFP4 exports.
        quant_config._megaquant_quantized_layers = (
            config.get("quantized_layers")
            or (config.get("quantization") or {}).get("quantized_layers")
            or {}
        )
''',
            "uniform NVFP4 layer metadata",
        )
    elif relative_path == "srt/configs/model_config.py":
        source = _replace_once(
            source,
            '''            self.hf_config.architectures[0] = "Qwen3_5ForCausalLMMTP"
            self.hf_config.num_nextn_predict_layers = 1
            self.hf_text_config.num_nextn_predict_layers = 1
''',
            '''            # Draft cache metadata must describe its one full-attention layer.
            # Copy before changing a config that may be shared with the target.
            self.hf_config = copy.deepcopy(self.hf_config)
            self.hf_text_config = get_hf_text_config(self.hf_config)
            self.hf_config.architectures[0] = "Qwen3_5ForCausalLMMTP"
            self.hf_config.num_nextn_predict_layers = 1
            self.hf_text_config.num_nextn_predict_layers = 1
            self.hf_text_config.num_hidden_layers = 1
            self.hf_text_config.layer_types = ["full_attention"]
            self.hf_text_config.full_attention_interval = 1
''',
            "MTP cache configuration",
        )
    else:
        raise PatchError(f"unknown patch target: {relative_path}")
    source = MARKER + source
    ast.parse(source, filename=relative_path)
    return source


def patch_tree(root: Path, version: str, *, check: bool = False) -> dict[str, Any]:
    """Preflight every file, then apply and record the complete patch."""
    if version != VERSION:
        raise PatchError(f"expected SGLang {VERSION}, found {version}; refusing to patch")
    manifest_path = root / MANIFEST
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            valid = (
                manifest["patch"] == PATCH_ID
                and manifest["sglang"] == VERSION
                and set(manifest["files"]) == set(UPSTREAM_SHA256)
                and all(
                    details["upstream_sha256"] == UPSTREAM_SHA256[name]
                    and details["patched_sha256"] == _sha256((root / name).read_bytes())
                    for name, details in manifest["files"].items()
                )
            )
        except (OSError, ValueError, KeyError, TypeError):
            valid = False
        if not valid:
            raise PatchError("existing Spark patch manifest or installed files were changed")
        return {"status": "already-patched", **manifest}

    pending: dict[str, bytes] = {}
    files: dict[str, dict[str, str]] = {}
    for relative_path, expected in UPSTREAM_SHA256.items():
        path = root / relative_path
        original = path.read_bytes()
        digest = _sha256(original)
        if digest != expected:
            raise PatchError(
                f"{relative_path}: upstream SHA256 mismatch ({digest}); refusing to patch"
            )
        patched = transform(relative_path, original.decode("utf-8")).encode("utf-8")
        pending[relative_path] = patched
        files[relative_path] = {
            "upstream_sha256": digest,
            "patched_sha256": _sha256(patched),
        }
    manifest = {"patch": PATCH_ID, "sglang": VERSION, "files": files}
    if not check:
        for relative_path, data in pending.items():
            (root / relative_path).write_bytes(data)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {"status": "checked" if check else "patched", **manifest}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify without changing files")
    args = parser.parse_args()
    try:
        version = importlib.metadata.version("sglang")
        spec = importlib.util.find_spec("sglang")
        if spec is None or spec.origin is None:
            raise PatchError("cannot locate the installed sglang package")
        result = patch_tree(Path(spec.origin).parent, version, check=args.check)
    except (PatchError, OSError, ValueError, importlib.metadata.PackageNotFoundError) as exc:
        parser.exit(1, f"Spark SGLang compatibility patch failed: {exc}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
