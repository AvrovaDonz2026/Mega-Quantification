import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("r6_runner_export_validation", HERE / "runner.py")
runner = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(runner)


def _write_safetensors(path: Path, keys: list[str]) -> None:
    header = {key: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]} for key in keys}
    payload = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(len(payload).to_bytes(8, "little") + payload + b"\0\0")


def _write_index(root: Path, weight_map: dict[str, str]) -> None:
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))


def test_repository_root_includes_runner_directory(tmp_path):
    for marker in ("pyproject.toml", "src", "recipes"):
        marker_path = tmp_path / marker
        marker_path.mkdir() if "." not in marker else marker_path.write_text("")
    assert runner._repository_root(tmp_path) == tmp_path


def test_export_index_validates_indexed_keys_against_safetensors_headers(tmp_path):
    _write_safetensors(tmp_path / "model-00001-of-00002.safetensors", ["layer.weight"])
    _write_safetensors(tmp_path / "model-00002-of-00002.safetensors", ["layer.bias"])
    _write_index(
        tmp_path,
        {
            "layer.weight": "model-00001-of-00002.safetensors",
            "layer.bias": "model-00002-of-00002.safetensors",
        },
    )

    report = runner._export_index_report(tmp_path)

    assert report["keys_validated"] is True
    assert report["validation"] == "safetensors-header-keys"
    assert report["header_keys_validated"] is True
    assert report["validated_key_count"] == 2
    assert report["missing_keys"] == []
    assert sorted(report["validated_shards"]) == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]


def test_export_index_reports_missing_header_key(tmp_path):
    _write_safetensors(tmp_path / "model.safetensors", ["actual.weight"])
    _write_index(tmp_path, {"expected.weight": "model.safetensors"})

    report = runner._export_index_report(tmp_path)

    assert report["keys_validated"] is False
    assert report["header_keys_validated"] is True
    assert report["missing_keys"] == ["expected.weight"]
    assert report["validated_key_count"] == 0


def test_export_index_rejects_unindexed_header_keys_and_path_escape(tmp_path):
    _write_safetensors(tmp_path / "model.safetensors", ["indexed.weight", "extra.weight"])
    _write_index(tmp_path, {"indexed.weight": "model.safetensors"})
    extra = runner._export_index_report(tmp_path)
    assert extra["keys_validated"] is False
    assert extra["unindexed_keys"] == ["model.safetensors:extra.weight"]

    outside = tmp_path.parent / "outside.safetensors"
    _write_safetensors(outside, ["layer.weight"])
    _write_index(tmp_path, {"layer.weight": "../outside.safetensors"})
    escaped = runner._export_index_report(tmp_path)
    assert escaped["keys_validated"] is False
    assert escaped["escaped_shards"] == ["../outside.safetensors"]


def test_export_index_fails_closed_for_unsupported_or_invalid_shards(tmp_path):
    (tmp_path / "model.bin").write_bytes(b"weights")
    _write_index(tmp_path, {"layer.weight": "model.bin"})
    unsupported = runner._export_index_report(tmp_path)
    assert unsupported["keys_validated"] is False
    assert unsupported["unsupported_shards"] == ["model.bin"]

    (tmp_path / "model.safetensors").write_bytes(b"\x01\x00")
    _write_index(tmp_path, {"layer.weight": "model.safetensors"})
    invalid = runner._export_index_report(tmp_path)
    assert invalid["keys_validated"] is False
    assert invalid["invalid_headers"] == ["model.safetensors"]
