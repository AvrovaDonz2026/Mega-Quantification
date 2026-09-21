"""llm-compressor / compressed-tensors backend (vLLM-oriented path).

This backend is import-safe without ``llmcompressor`` or ``compressed_tensors``
installed. ``available()`` / ``supports()`` / ``describe_scheme()`` never import
those packages. ``build_recipe()`` and ``quantize()`` import them lazily.

Runtime mapping
---------------
vLLM native support is strongest for:

- NVFP4 W4A4 (stock compressed-tensors ``NVFP4``)
- NVFP4A16 (stock ``NVFP4A16``)
- FP8 W8A8 (stock ``FP8_DYNAMIC``)

**NVFP4 W4A8** (FP4 weights + FP8 activations) is officially a
ModelOpt / TensorRT-LLM path (``W4A8_NVFP4_FP8`` / ``nvfp4_bs32``). There is
**no** stock ``NVFP4A8`` preset in compressed-tensors, so this backend emits a
custom :class:`~compressed_tensors.quantization.quant_scheme.QuantizationScheme`:

- weights: FP4, ``TENSOR_GROUP``, ``group_size=32``, FP8 E4M3 scales
- input activations: FP8 E4M3, ``TOKEN``, ``dynamic=True`` (default)

The resulting compressed-tensors checkpoint may not have a fused vLLM kernel
for FP4 weights with FP8 activations. Prefer ``backend=modelopt`` for W4A8
production exports.

oneshot calibration
-------------------
``oneshot()`` prefers a HuggingFace ``Dataset``. If ``calib_iter`` is already a
tokenized HF Dataset / DatasetDict / DataLoader it is passed through. A plain
Python iterator is materialized with :func:`iterator_to_hf_dataset` (uses
``datasets.Dataset.from_list`` when ``datasets`` is installed, otherwise a
list of row dicts).
"""

from __future__ import annotations

import inspect
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from megaquant.schemes.catalog import SCHEME_CATALOG, get_scheme
from megaquant.schemes.nvfp4 import DEFAULT_LINEAR_TARGETS

__all__ = [
    "LLMCompressorBackend",
    "iterator_to_hf_dataset",
]

# compressed-tensors preset names for catalog keys that have a stock scheme.
_STOCK_PRESETS: dict[str, str] = {
    "nvfp4_w4a4": "NVFP4",
    "nvfp4_w4a16": "NVFP4A16",
    "fp8_w8a8": "FP8_DYNAMIC",
}

_SCALE_DTYPE_ALIASES = frozenset(
    {"float8_e4m3fn", "fp8", "e4m3", "float8_e4m3", "torch.float8_e4m3fn"}
)


# ---------------------------------------------------------------------------
# Backend protocol (duck-typed if Agent ModelOpt has not landed backends.base)
# ---------------------------------------------------------------------------

try:
    from megaquant.backends.base import QuantBackend as QuantBackend
except ImportError:

    @runtime_checkable
    class QuantBackend(Protocol):  # type: ignore[no-redef]
        name: str

        def available(self) -> bool: ...

        def supports(self, scheme_name: str) -> bool: ...

        def quantize(self, model: Any, resolved: Any, calib_iter: Any) -> Any: ...

        def export(self, model: Any, recipe: Any, tokenizer: Any = None) -> Path: ...


def _get(obj: Any, *names: str, default: Any = None) -> Any:
    """Duck-typed attribute / mapping lookup. Skips missing and ``None`` values."""
    if obj is None:
        return default
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            value = obj[name]
            if value is not None:
                return value
            continue
        if not isinstance(obj, Mapping) and hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return default


def _from_plan_or_recipe(plan: Any, *names: str, default: Any = None) -> Any:
    """Look up ``names`` on a plan, then on ``plan.recipe`` (ResolvedPlan)."""
    value = _get(plan, *names, default=None)
    if value is not None:
        return value
    recipe = _get(plan, "recipe")
    if recipe is not None and recipe is not plan:
        value = _get(recipe, *names, default=None)
        if value is not None:
            return value
    return default


def _as_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        dumped = obj.model_dump()
        if isinstance(dumped, dict):
            return dumped
    if is_dataclass(obj) and not isinstance(obj, type):
        from dataclasses import asdict

        return asdict(obj)
    if hasattr(obj, "dict") and callable(obj.dict):
        try:
            dumped = obj.dict()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass
    out: dict[str, Any] = {}
    for key in ("name", "targets", "weights", "activations", "ignore"):
        if hasattr(obj, key):
            out[key] = getattr(obj, key)
    return out


def _scheme_name(plan: Any) -> str:
    if isinstance(plan, str):
        return plan
    scheme = _from_plan_or_recipe(plan, "scheme")
    if isinstance(scheme, str):
        return scheme
    name = _get(scheme, "name")
    if isinstance(name, str):
        return name
    # SchemeDef itself (has .name, no .scheme)
    fallback = _get(plan, "name")
    if isinstance(fallback, str) and fallback in SCHEME_CATALOG:
        return fallback
    raise ValueError(f"Cannot resolve scheme name from plan: {type(plan)!r}")


def _resolved_groups(plan: Any) -> list[dict[str, Any]]:
    # Prefer ResolvedPlan.groups (already catalog + recipe overrides).
    groups = _get(plan, "groups")
    if not groups:
        groups = _get(_get(plan, "recipe"), "groups")
    if groups:
        return [_as_dict(group) for group in groups]
    return [dict(group) for group in get_scheme(_scheme_name(plan)).groups]


def _ignore_list(plan: Any) -> list[str]:
    ignore = _get(plan, "ignore", "ignores", default=None) or []
    extra = _from_plan_or_recipe(plan, "extra_ignore", default=None) or []
    out: list[str] = []
    for item in list(ignore) + list(extra):
        if item not in out:
            out.append(item)
    return out


def _kv_cache_name(plan: Any) -> str | None:
    kv = _from_plan_or_recipe(plan, "kv_cache")
    if kv is None:
        try:
            kv = get_scheme(_scheme_name(plan)).kv_cache
        except (KeyError, ValueError):
            kv = None
    if kv is False:
        return None
    if kv is True:
        return "fp8"
    if isinstance(kv, str):
        return kv
    return None


def _group_targets(group: Mapping[str, Any]) -> list[str]:
    targets = group.get("targets") or list(DEFAULT_LINEAR_TARGETS)
    if isinstance(targets, str):
        return [targets]
    return list(targets)


def _is_dynamic_token(spec: Any) -> bool:
    data = _as_dict(spec)
    if not data:
        return False
    dynamic = data.get("dynamic", False)
    is_dynamic = dynamic is True or str(dynamic).lower() == "true"
    strategy = str(data.get("strategy") or "token").lower()
    return is_dynamic and strategy == "token"


def _uniform_stock_ok(scheme_name: str, groups: Sequence[Mapping[str, Any]]) -> bool:
    """True when a single group matches the stock preset for ``scheme_name``."""
    if scheme_name not in _STOCK_PRESETS or len(groups) != 1:
        return False
    group = groups[0]
    weights = _as_dict(group.get("weights"))
    acts = group.get("activations")
    if scheme_name == "nvfp4_w4a4":
        return (
            int(weights.get("bits") or 0) == 4
            and int(weights.get("group_size") or 0) == 16
            and str(weights.get("strategy") or "") == "tensor_group"
            and acts is not None
            and int(_as_dict(acts).get("bits") or 0) == 4
        )
    if scheme_name == "nvfp4_w4a16":
        return (
            int(weights.get("bits") or 0) == 4
            and int(weights.get("group_size") or 0) == 16
            and acts is None
        )
    if scheme_name == "fp8_w8a8":
        return (
            int(weights.get("bits") or 0) == 8
            and _is_dynamic_token(acts)
            and str(weights.get("strategy") or "channel").lower() in {"channel", ""}
        )
    return False


def _output_dir(recipe: Any) -> str:
    if isinstance(recipe, (str, Path)):
        return str(recipe)
    export = _get(recipe, "export")
    for candidate in (
        _get(export, "output_dir", "output"),
        _get(recipe, "output_dir", "output"),
    ):
        if candidate:
            return str(candidate)
    raise ValueError("Cannot determine export output_dir from recipe")


def _calibration_kwargs(plan: Any) -> dict[str, Any]:
    calib = _from_plan_or_recipe(plan, "calibration")
    max_seq = _get(calib, "max_seq_length", default=2048)
    num_samples = _get(calib, "num_samples", default=512)
    batch_size = _get(calib, "batch_size", default=1)
    text_field = _get(calib, "text_field", default="text")
    return {
        "max_seq_length": max_seq,
        "num_calibration_samples": num_samples,
        "batch_size": batch_size,
        "text_column": text_field or "text",
    }


def _to_python(value: Any) -> Any:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach()
    if hasattr(value, "cpu") and callable(value.cpu):
        try:
            value = value.cpu()
        except Exception:
            pass
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return value.tolist()
        except Exception:
            pass
    return value


def _normalize_row(item: Any, text_field: str) -> dict[str, Any]:
    if isinstance(item, str):
        return {text_field: item}
    if isinstance(item, Mapping):
        return {str(key): _to_python(val) for key, val in item.items()}
    if hasattr(item, "keys") and hasattr(item, "__getitem__"):
        try:
            return {str(key): _to_python(item[key]) for key in item.keys()}
        except Exception:
            pass
    if hasattr(item, "input_ids"):
        row: dict[str, Any] = {"input_ids": _to_python(item.input_ids)}
        if hasattr(item, "attention_mask"):
            row["attention_mask"] = _to_python(item.attention_mask)
        return row
    return {text_field: item}


def _dataset_has_column(dataset: Any, column: str) -> bool:
    if dataset is None or not column:
        return False
    if isinstance(dataset, str):
        return True
    names = getattr(dataset, "column_names", None)
    if names is not None:
        return column in names
    if isinstance(dataset, Mapping):
        return column in dataset
    if isinstance(dataset, (list, tuple)) and dataset:
        first = dataset[0]
        if isinstance(first, Mapping):
            return column in first
        if hasattr(first, "keys"):
            try:
                return column in first.keys()
            except Exception:
                return False
    return False


def _is_hf_dataset(obj: Any) -> bool:
    if obj is None:
        return False
    module = type(obj).__module__ or ""
    if module.startswith("datasets"):
        return True
    try:
        from datasets import Dataset, DatasetDict, IterableDataset

        return isinstance(obj, (Dataset, DatasetDict, IterableDataset))
    except ImportError:
        return False


def iterator_to_hf_dataset(
    calib_iter: Any,
    *,
    text_field: str = "text",
) -> Any:
    """Materialize a calibration iterator into something ``oneshot`` can consume.

    ``llmcompressor.oneshot`` accepts a HuggingFace Dataset / DatasetDict, a
    dataset id string, or (recent versions) a PyTorch DataLoader. A plain
    Python iterator is **not** a documented oneshot input, so this helper
    converts rows via ``datasets.Dataset.from_list`` when ``datasets`` is
    installed. If ``datasets`` is missing the rows are returned as a ``list``
    of dicts — the caller should still prefer installing ``datasets``.

    Already-tokenized HuggingFace Datasets and DataLoaders are passed through.
    """
    if calib_iter is None:
        return None
    if isinstance(calib_iter, str):
        return calib_iter
    if _is_hf_dataset(calib_iter):
        return calib_iter
    try:
        from torch.utils.data import DataLoader

        if isinstance(calib_iter, DataLoader):
            return calib_iter
    except ImportError:
        pass

    if isinstance(calib_iter, (list, tuple)):
        rows = [_normalize_row(item, text_field) for item in calib_iter]
    elif isinstance(calib_iter, (Iterator, Iterable)) and not isinstance(
        calib_iter, (str, bytes, Mapping)
    ):
        rows = [_normalize_row(item, text_field) for item in calib_iter]
    else:
        rows = [_normalize_row(calib_iter, text_field)]

    try:
        from datasets import Dataset

        return Dataset.from_list(rows)
    except ImportError:
        return rows


def _call_oneshot(oneshot: Any, kwargs: dict[str, Any]) -> Any:
    """Invoke ``oneshot`` with only kwargs its signature accepts.

    Typical supported names: ``model``, ``dataset``, ``recipe``,
    ``max_seq_length``, ``num_calibration_samples``. ``save=False`` is passed
    when that parameter exists; otherwise ``output_dir=None`` is used so
    ``quantize()`` does not write a checkpoint (``export()`` does).
    """
    typical = (
        "model",
        "dataset",
        "recipe",
        "max_seq_length",
        "num_calibration_samples",
        "batch_size",
        "text_column",
        "save",
        "output_dir",
    )
    var_kw_safe = (
        "model",
        "dataset",
        "recipe",
        "max_seq_length",
        "num_calibration_samples",
        "batch_size",
        "text_column",
        "output_dir",
    )
    try:
        signature = inspect.signature(oneshot)
        named = {
            name
            for name, param in signature.parameters.items()
            if param.kind != inspect.Parameter.VAR_KEYWORD
        }
        has_var = any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()
        )
    except (TypeError, ValueError):
        named = set(typical)
        has_var = False

    payload: dict[str, Any] = {}
    for key, value in kwargs.items():
        if key in named:
            payload[key] = value
        elif has_var and key in var_kw_safe:
            payload[key] = value
    return oneshot(**payload)


class LLMCompressorBackend:
    """QuantBackend that emits compressed-tensors via llm-compressor oneshot."""

    name = "llmcompressor"

    def available(self) -> bool:
        try:
            import compressed_tensors  # noqa: F401
            import llmcompressor  # noqa: F401
        except ImportError:
            return False
        return True

    def supports(self, scheme_name: str) -> bool:
        if not scheme_name:
            return False
        try:
            get_scheme(scheme_name)
        except KeyError:
            return False
        return True

    def describe_scheme(
        self,
        scheme_name: str | None = None,
        plan: Any = None,
    ) -> dict[str, Any]:
        """JSON description for ``--dry-run``. Never imports llm-compressor."""
        name = scheme_name
        if name is None and plan is not None:
            name = _scheme_name(plan)
        if name is None:
            name = "nvfp4_w4a8"
        scheme = get_scheme(name)
        groups = _resolved_groups(plan) if plan is not None else list(scheme.groups)
        ignore = _ignore_list(plan) if plan is not None else []
        stock = _STOCK_PRESETS.get(name)
        custom = name in {"nvfp4_w4a8", "nvfp4_mixed"} or (
            stock is not None and not _uniform_stock_ok(name, groups)
        )
        payload: dict[str, Any] = {
            "backend": self.name,
            "available": self.available(),
            "scheme": scheme.name,
            "groups": groups,
            "ignore": ignore,
            "kv_cache": _kv_cache_name(plan) if plan is not None else scheme.kv_cache,
            "notes": scheme.notes,
            "vllm_support": scheme.vllm_support,
            "trtllm_support": scheme.trtllm_support,
            "stock_preset": None if custom else stock,
            "custom_quantization_scheme": custom,
            "first_class_w4a8": "modelopt",
        }
        if name == "nvfp4_w4a8":
            payload["quantization_args"] = {
                "weights": {
                    "num_bits": 4,
                    "type": "FLOAT",
                    "strategy": "TENSOR_GROUP",
                    "symmetric": True,
                    "dynamic": False,
                    "group_size": 32,
                    "scale_dtype": "FP8_E4M3_DATA.dtype",
                    "zp_dtype": "FP8_E4M3_DATA.dtype",
                },
                "input_activations": {
                    "num_bits": 8,
                    "type": "FLOAT",
                    "strategy": "TOKEN",
                    "symmetric": True,
                    "dynamic": True,
                    "observer": None,
                },
            }
        return payload

    def build_recipe(self, plan: Any) -> Any:
        """Construct a ``QuantizationModifier`` (lazy compressed-tensors import)."""
        self._require_available()
        (
            QuantizationArgs,
            QuantizationScheme,
            QuantizationModifier,
            enums,
        ) = self._lazy_quant_types()

        scheme_name = _scheme_name(plan)
        groups = _resolved_groups(plan)
        ignore = _ignore_list(plan)
        kv_cache_scheme = self._build_kv_cache_scheme(plan, QuantizationArgs, enums)

        modifier_kwargs: dict[str, Any] = {"ignore": ignore}
        if kv_cache_scheme is not None:
            modifier_kwargs["kv_cache_scheme"] = kv_cache_scheme

        observer = self._weight_observer(plan)
        if observer is not None:
            modifier_kwargs["weight_observer"] = observer

        if scheme_name in {"nvfp4_w4a8", "nvfp4_mixed"} or len(groups) > 1:
            config_groups: dict[str, Any] = {}
            for group in groups:
                name = str(group.get("name") or f"group_{len(config_groups)}")
                config_groups[name] = self._group_to_scheme(
                    group, QuantizationArgs, QuantizationScheme, enums
                )
            return QuantizationModifier(config_groups=config_groups, **modifier_kwargs)

        if (
            scheme_name in _STOCK_PRESETS
            and _uniform_stock_ok(scheme_name, groups)
            and self._preset_exists(_STOCK_PRESETS[scheme_name])
        ):
            targets = _group_targets(groups[0]) if groups else list(DEFAULT_LINEAR_TARGETS)
            return QuantizationModifier(
                targets=targets,
                scheme=_STOCK_PRESETS[scheme_name],
                **modifier_kwargs,
            )

        # Fallback: explicit QuantizationScheme (e.g. NVFP4 preset missing, or
        # static FP8, or a group_size override).
        if scheme_name == "nvfp4_w4a4":
            config_groups = {
                "nvfp4_w4a4": self._nvfp4_w4a4_scheme(
                    groups, QuantizationArgs, QuantizationScheme, enums
                )
            }
            return QuantizationModifier(config_groups=config_groups, **modifier_kwargs)

        config_groups = {}
        for group in groups:
            name = str(group.get("name") or f"group_{len(config_groups)}")
            config_groups[name] = self._group_to_scheme(
                group, QuantizationArgs, QuantizationScheme, enums
            )
        return QuantizationModifier(config_groups=config_groups, **modifier_kwargs)

    def build_quant_cfg(self, plan: Any) -> Any:
        """``QuantBackend`` alias for :meth:`build_recipe`."""
        return self.build_recipe(plan)

    def quantize(self, model: Any, plan: Any, calib_iter: Any) -> Any:
        """Run ``llmcompressor.oneshot`` in-place; does not write a checkpoint."""
        self._require_available()
        from llmcompressor import oneshot

        modifier = self.build_recipe(plan)
        calib_kwargs = _calibration_kwargs(plan)
        dataset = calib_iter
        already_dataset = isinstance(dataset, str) or _is_hf_dataset(dataset)
        if dataset is not None and not already_dataset:
            dataset = iterator_to_hf_dataset(
                calib_iter, text_field=str(calib_kwargs.get("text_column") or "text")
            )

        call = {
            "model": model,
            "dataset": dataset,
            "recipe": modifier,
            "max_seq_length": calib_kwargs["max_seq_length"],
            "num_calibration_samples": calib_kwargs["num_calibration_samples"],
            "batch_size": calib_kwargs["batch_size"],
            "text_column": calib_kwargs["text_column"],
            "save": False,
            "output_dir": None,
        }
        # Pipeline calib iter is already tokenized (input_ids / attention_mask).
        # oneshot looking for text_column="text" would KeyError.
        if not _dataset_has_column(dataset, str(calib_kwargs.get("text_column") or "text")):
            call.pop("text_column", None)
        return _call_oneshot(oneshot, call)

    def export(self, model: Any, recipe: Any, tokenizer: Any = None) -> Path:
        """Save a compressed HuggingFace checkpoint (``save_compressed=True``)."""
        output_dir = Path(_output_dir(recipe))
        output_dir.mkdir(parents=True, exist_ok=True)
        save = getattr(model, "save_pretrained", None)
        if save is None:
            raise TypeError("model has no save_pretrained(); cannot export")
        try:
            save(str(output_dir), save_compressed=True)
        except TypeError:
            save(str(output_dir))
        if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(str(output_dir))
        return output_dir

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_available(self) -> None:
        if not self.available():
            raise RuntimeError(
                "llmcompressor backend is not installed. Install with: "
                "pip install 'megaquant[llmcompressor]'"
            )

    @staticmethod
    def _lazy_quant_types() -> tuple[Any, Any, Any, dict[str, Any]]:
        from compressed_tensors.quantization.quant_args import (
            FP8_E4M3_DATA,
            DynamicType,
            QuantizationArgs,
            QuantizationStrategy,
            QuantizationType,
        )
        from compressed_tensors.quantization.quant_scheme import QuantizationScheme
        from llmcompressor.modifiers.quantization import QuantizationModifier

        return (
            QuantizationArgs,
            QuantizationScheme,
            QuantizationModifier,
            {
                "FP8_E4M3_DATA": FP8_E4M3_DATA,
                "DynamicType": DynamicType,
                "QuantizationStrategy": QuantizationStrategy,
                "QuantizationType": QuantizationType,
            },
        )

    @staticmethod
    def _preset_exists(name: str) -> bool:
        try:
            from compressed_tensors.quantization.quant_scheme import is_preset_scheme

            return bool(is_preset_scheme(name))
        except Exception:
            return False

    @staticmethod
    def _weight_observer(plan: Any) -> str | None:
        algorithm = _from_plan_or_recipe(plan, "algorithm")
        if not algorithm or not isinstance(algorithm, str):
            return None
        mapping = {"max": "minmax", "mse": "mse"}
        return mapping.get(algorithm.lower())

    def _build_kv_cache_scheme(
        self,
        plan: Any,
        QuantizationArgs: Any,
        enums: Mapping[str, Any],
    ) -> Any | None:
        name = _kv_cache_name(plan)
        if not name:
            return None
        existing = _get(plan, "kv_cache_scheme")
        if existing is not None and not isinstance(existing, str):
            return existing
        lowered = str(name).lower()
        if lowered not in {"fp8", "float8", "float8_e4m3fn", "e4m3"}:
            return None
        return QuantizationArgs(
            num_bits=8,
            type=enums["QuantizationType"].FLOAT,
            strategy=enums["QuantizationStrategy"].TENSOR,
            symmetric=True,
            dynamic=False,
        )

    def _group_to_scheme(
        self,
        group: Mapping[str, Any],
        QuantizationArgs: Any,
        QuantizationScheme: Any,
        enums: Mapping[str, Any],
    ) -> Any:
        """Build a QuantizationScheme, using exact W4A8 args for nvfp4 gs=32."""
        weights = _as_dict(group.get("weights"))
        fmt = str(weights.get("format") or "")
        group_size = weights.get("group_size")
        if fmt == "nvfp4" and group_size == 32:
            acts = group.get("activations")
            if acts is not None and not _is_dynamic_token(acts) and _as_dict(acts):
                input_activations = self._quant_args_from_spec(acts, QuantizationArgs, enums)
            else:
                input_activations = self._nvfp4_w4a8_activation_args(QuantizationArgs, enums)
            return QuantizationScheme(
                targets=_group_targets(group),
                weights=self._nvfp4_w4a8_weight_args(QuantizationArgs, enums),
                input_activations=input_activations,
            )
        return self._scheme_from_group(group, QuantizationArgs, QuantizationScheme, enums)

    @staticmethod
    def _nvfp4_w4a8_weight_args(QuantizationArgs: Any, enums: Mapping[str, Any]) -> Any:
        fp8 = enums["FP8_E4M3_DATA"].dtype
        return QuantizationArgs(
            num_bits=4,
            type=enums["QuantizationType"].FLOAT,
            strategy=enums["QuantizationStrategy"].TENSOR_GROUP,
            symmetric=True,
            dynamic=False,
            group_size=32,
            scale_dtype=fp8,
            zp_dtype=fp8,
        )

    @staticmethod
    def _nvfp4_w4a8_activation_args(QuantizationArgs: Any, enums: Mapping[str, Any]) -> Any:
        return QuantizationArgs(
            num_bits=8,
            type=enums["QuantizationType"].FLOAT,
            strategy=enums["QuantizationStrategy"].TOKEN,
            symmetric=True,
            dynamic=True,
            observer=None,
        )

    def _nvfp4_w4a4_scheme(
        self,
        groups: Sequence[Mapping[str, Any]],
        QuantizationArgs: Any,
        QuantizationScheme: Any,
        enums: Mapping[str, Any],
    ) -> Any:
        targets = _group_targets(groups[0]) if groups else list(DEFAULT_LINEAR_TARGETS)
        fp8 = enums["FP8_E4M3_DATA"].dtype
        local = enums["DynamicType"].LOCAL
        return QuantizationScheme(
            targets=targets,
            weights=QuantizationArgs(
                num_bits=4,
                type=enums["QuantizationType"].FLOAT,
                strategy=enums["QuantizationStrategy"].TENSOR_GROUP,
                symmetric=True,
                dynamic=False,
                group_size=16,
                scale_dtype=fp8,
                zp_dtype=fp8,
            ),
            input_activations=QuantizationArgs(
                num_bits=4,
                type=enums["QuantizationType"].FLOAT,
                strategy=enums["QuantizationStrategy"].TENSOR_GROUP,
                symmetric=True,
                dynamic=local,
                group_size=16,
                observer="static_minmax",
                scale_dtype=fp8,
                zp_dtype=fp8,
            ),
        )

    def _scheme_from_group(
        self,
        group: Mapping[str, Any],
        QuantizationArgs: Any,
        QuantizationScheme: Any,
        enums: Mapping[str, Any],
    ) -> Any:
        weights_spec = group.get("weights")
        acts_spec = group.get("activations")
        return QuantizationScheme(
            targets=_group_targets(group),
            weights=self._quant_args_from_spec(weights_spec, QuantizationArgs, enums),
            input_activations=(
                None
                if acts_spec is None
                else self._quant_args_from_spec(acts_spec, QuantizationArgs, enums)
            ),
        )

    def _quant_args_from_spec(
        self,
        spec: Any,
        QuantizationArgs: Any,
        enums: Mapping[str, Any],
    ) -> Any | None:
        data = _as_dict(spec)
        if not data:
            return None

        bits = int(data.get("bits") or data.get("num_bits") or 8)
        fmt = str(data.get("format") or "").lower()
        qtype = enums["QuantizationType"].FLOAT
        if fmt in {"int4", "int8", "int"} or str(data.get("type") or "").lower() == "int":
            qtype = enums["QuantizationType"].INT
        elif str(data.get("type") or "").lower() == "float":
            qtype = enums["QuantizationType"].FLOAT
        elif fmt in {"nvfp4", "fp8", "mxfp4", "float"}:
            qtype = enums["QuantizationType"].FLOAT

        strategy_raw = data.get("strategy")
        strategy = None
        if strategy_raw is not None:
            strategy = enums["QuantizationStrategy"](str(strategy_raw).lower())

        dynamic = data.get("dynamic", False)
        if isinstance(dynamic, str) and dynamic.lower() == "local":
            dynamic = enums["DynamicType"].LOCAL
        elif isinstance(dynamic, str):
            dynamic = dynamic.lower() in {"true", "1", "yes"}

        kwargs: dict[str, Any] = {
            "num_bits": bits,
            "type": qtype,
            "symmetric": bool(data.get("symmetric", True)),
            "dynamic": dynamic,
        }
        if strategy is not None:
            kwargs["strategy"] = strategy
        group_size = data.get("group_size")
        if group_size is not None:
            kwargs["group_size"] = int(group_size)

        scale_dtype = data.get("scale_dtype")
        fp8 = enums["FP8_E4M3_DATA"].dtype
        if scale_dtype is not None and str(scale_dtype).lower() in _SCALE_DTYPE_ALIASES:
            kwargs["scale_dtype"] = fp8
        elif bits == 4 and qtype == enums["QuantizationType"].FLOAT:
            kwargs["scale_dtype"] = fp8
            kwargs["zp_dtype"] = fp8

        zp = data.get("zp_dtype")
        if zp is not None and str(zp).lower() in _SCALE_DTYPE_ALIASES:
            kwargs["zp_dtype"] = fp8
        elif bits == 4 and qtype == enums["QuantizationType"].FLOAT:
            kwargs["zp_dtype"] = fp8

        if "observer" in data:
            kwargs["observer"] = data["observer"]

        return QuantizationArgs(**kwargs)


def _try_register_backend() -> None:
    try:
        from megaquant.registry import register_backend
    except ImportError:
        return
    backend = LLMCompressorBackend()
    try:
        signature = inspect.signature(register_backend)
        params = [
            p
            for p in signature.parameters.values()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
            and p.name not in {"self", "cls"}
        ]
    except (TypeError, ValueError):
        params = []
    try:
        if len(params) >= 2:
            register_backend(backend.name, backend)
        else:
            register_backend(backend)
    except TypeError:
        try:
            register_backend(backend.name, LLMCompressorBackend)
        except Exception:
            return
    except Exception:
        return


_try_register_backend()
