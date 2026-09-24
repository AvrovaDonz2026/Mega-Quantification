"""Recipe YAML shape — no pydantic / GPU / Hub required."""

from __future__ import annotations

from pathlib import Path

import yaml

REQUIRED_TOP_KEYS = (
    "name",
    "model",
    "backend",
    "scheme",
    "algorithm",
    "kv_cache",
    "family",
    "extra_ignore",
    "calibration",
    "export",
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RECIPES = REPO_ROOT / "recipes"
W4A8 = RECIPES / "qwen3.8-27b-nvfp4-w4a8.yaml"
MIXED = RECIPES / "qwen3.8-27b-nvfp4-mixed.yaml"
W4A4 = RECIPES / "qwen3.8-27b-nvfp4-w4a4.yaml"
W4A4_5090 = RECIPES / "qwen3.8-27b-nvfp4-w4a4.5090.yaml"
W4A4_PUBLIC = RECIPES / "qwen3.8-27b-nvfp4-w4a4.public-calib.yaml"
MIXED_5090 = RECIPES / "qwen3.8-27b-nvfp4-mixed.5090.yaml"
MIXED_PUBLIC = RECIPES / "qwen3.8-27b-nvfp4-mixed.public-calib.yaml"
GPQA = RECIPES / "eval-gpqa-diamond.yaml"
GPQA_5090 = RECIPES / "eval-gpqa-diamond.5090.yaml"
GPQA_6000D = RECIPES / "eval-gpqa-diamond.6000d.yaml"
LOCAL_EXPORT = "outputs/Qwen3.8-27B-NVFP4-W4A8"
HUB_BF16 = "Qwen/Qwen3.8-27B"


def _load(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text())
    assert isinstance(payload, dict), f"{path} must be a mapping"
    return payload


def _assert_schema(data: dict, path: Path) -> None:
    missing = [key for key in REQUIRED_TOP_KEYS if key not in data]
    assert not missing, f"{path.name} missing keys: {missing}"
    assert "source" in data["model"]
    assert "output_dir" in data["export"]
    assert "num_samples" in data["calibration"]
    assert "dataset" in data["calibration"]


def test_w4a8_recipe_yaml_shape_and_source() -> None:
    data = _load(W4A8)
    _assert_schema(data, W4A8)
    assert data["model"]["source"] == "Qwen/Qwen3.8-27B"
    assert data["scheme"] == "nvfp4_w4a8"
    assert data["backend"] == "modelopt"
    assert data["algorithm"] == "max"
    assert data["kv_cache"] == "fp8"
    assert data["family"] == "qwen3_5"
    assert data["calibration"]["num_samples"] == 512
    assert data["calibration"]["dataset"] == "nvidia/Nemotron-Post-Training-Dataset-v2"
    assert data["export"]["output_dir"] == "outputs/Qwen3.8-27B-NVFP4-W4A8"
    assert data["model"]["quantize_vision"] is False
    assert data["model"]["quantize_mtp"] is False


def test_w4a8_recipe_documents_sglang_mixed() -> None:
    text = W4A8.read_text()
    lowered = text.lower()
    assert "mixed" in lowered
    assert "16" in text
    assert "sglang" in lowered or "mixed_precision" in lowered
    assert "trtllm.yaml" not in lowered
    assert "not" in lowered and "block 32" in lowered


def test_trtllm_uniform_w4a8_recipe_is_not_shipped() -> None:
    assert not (RECIPES / "qwen3.8-27b-nvfp4-w4a8-trtllm.yaml").exists()
    for path in RECIPES.glob("*.yaml"):
        data = _load(path)
        assert data.get("scheme") != "w4a8_nvfp4_fp8", path.name


def test_mixed_recipe_matches_nvidia_intent() -> None:
    data = _load(MIXED)
    _assert_schema(data, MIXED)
    assert data["model"]["source"] == "Qwen/Qwen3.8-27B"
    assert data["scheme"] == "nvfp4_mixed"
    assert data["algorithm"] == "local_hessian"
    assert data["calibration"]["num_samples"] == 2048
    assert data["calibration"]["batch_size"] == 1
    assert data["calibration"]["dataset"] == "nvidia/Nemotron-Post-Training-Dataset-v3"
    assert data["export"]["output_dir"] == "outputs/Qwen3.8-27B-NVFP4-mixed"
    assert data["family"] == "qwen3_5"
    text = MIXED.read_text().lower()
    assert "16" in text
    assert "group_size" in text or "group size" in text


def test_w4a4_comparison_recipe_if_present() -> None:
    if not W4A4.is_file():
        return
    data = _load(W4A4)
    _assert_schema(data, W4A4)
    assert data["model"]["source"] == "Qwen/Qwen3.8-27B"
    assert data["scheme"] == "nvfp4_w4a4"
    text = W4A4.read_text().lower()
    assert "16" in text
    assert "block" in text or "group" in text


def test_5090_and_public_calib_recipes() -> None:
    for path in (W4A4_5090, W4A4_PUBLIC, MIXED_5090, MIXED_PUBLIC):
        assert path.is_file(), path.name
        data = _load(path)
        _assert_schema(data, path)
        assert data["model"]["source"] == "Qwen/Qwen3.8-27B"
        assert data["backend"] == "modelopt"
        assert data["family"] == "qwen3_5"
        assert data["calibration"]["dataset"] == "HuggingFaceH4/ultrachat_200k"
        assert data["algorithm"] == "max"

    w4a4_5090 = _load(W4A4_5090)
    assert w4a4_5090["scheme"] == "nvfp4_w4a4"
    assert w4a4_5090["calibration"]["num_samples"] == 256
    assert w4a4_5090["calibration"]["max_seq_length"] == 1024
    assert w4a4_5090["calibration"]["batch_size"] == 4
    assert w4a4_5090["export"]["output_dir"] == "outputs/Qwen3.8-27B-NVFP4-W4A4"

    mixed_5090 = _load(MIXED_5090)
    assert mixed_5090["scheme"] == "nvfp4_mixed"
    assert mixed_5090["algorithm"] == "max"
    assert mixed_5090["algorithm"] != "local_hessian"
    assert mixed_5090["calibration"]["num_samples"] == 256
    assert mixed_5090["calibration"]["max_seq_length"] == 1024
    assert mixed_5090["calibration"]["batch_size"] == 4
    assert mixed_5090["export"]["output_dir"] == "outputs/Qwen3.8-27B-NVFP4-W4A8"

    w4a4_pub = _load(W4A4_PUBLIC)
    assert w4a4_pub["scheme"] == "nvfp4_w4a4"
    assert w4a4_pub["calibration"]["num_samples"] == 512
    assert w4a4_pub["calibration"]["max_seq_length"] == 2048
    assert w4a4_pub["calibration"]["batch_size"] == 1
    w4a16 = RECIPES / "qwen3.8-27b-nvfp4-w4a16-mixed.5090.yaml"
    assert w4a16.is_file()
    w4a16_data = _load(w4a16)
    _assert_schema(w4a16_data, w4a16)
    assert w4a16_data["scheme"] == "nvfp4_w4a16_mixed"
    assert w4a16_data["algorithm"] == "max"
    assert w4a16_data["calibration"]["num_samples"] == 256
    assert w4a16_data["calibration"]["max_seq_length"] == 1024
    assert w4a16_data["export"]["output_dir"] == "outputs/Qwen3.8-27B-NVFP4-W4A16-mixed"

    mixed_pub = _load(MIXED_PUBLIC)
    assert mixed_pub["scheme"] == "nvfp4_mixed"
    assert mixed_pub["algorithm"] == "max"
    assert mixed_pub["algorithm"] != "local_hessian"
    assert mixed_pub["calibration"]["num_samples"] == 512
    assert mixed_pub["calibration"]["max_seq_length"] == 2048
    assert mixed_pub["calibration"]["batch_size"] == 1
    assert mixed_pub["export"]["output_dir"] == "outputs/Qwen3.8-27B-NVFP4-mixed"


def test_mixed_public_calib_docs_match_max_512() -> None:
    """Runbook and recipe index must describe public-calib as max / 512, not Hessian 2048."""
    index = (RECIPES / "README.md").read_text()
    runbook = (REPO_ROOT / "docs" / "qwen3.8-27b.md").read_text()
    for blob, label in (
        (index, "recipes/README.md"),
        (runbook, "docs/qwen3.8-27b.md"),
    ):
        rows = [line for line in blob.splitlines() if "mixed.public-calib.yaml" in line]
        assert rows, f"{label} must mention mixed.public-calib.yaml"
        for line in rows:
            lowered = line.lower()
            assert "local_hessian" not in lowered, label
            assert "local-hessian" not in lowered, label
            assert "max" in lowered, label
            assert "512" in line, label
            assert "2048" not in line, label


def test_mixed_5090_docs_agree_on_w4a8_output() -> None:
    """Production mixed.5090 export is the W4A8 directory used by restore-mtp / eval."""
    runbook = (REPO_ROOT / "docs" / "qwen3.8-27b.md").read_text()
    index = (RECIPES / "README.md").read_text()
    rows = [line for line in runbook.splitlines() if "mixed.5090.yaml" in line]
    assert rows
    assert any("Qwen3.8-27B-NVFP4-W4A8" in line for line in rows)
    assert "Qwen3.8-27B-NVFP4-W4A8" in index
    skill = (REPO_ROOT / "SKILL.md").read_text()
    assert "qwen3.8-27b-nvfp4-mixed.5090.yaml" in skill
    assert "outputs/Qwen3.8-27B-NVFP4-W4A8" in skill


def _assert_gpqa_official_cards(
    data: dict,
    *,
    attention_backend: str,
    kv_offloading_size_gb: float = 12,
    disable_cuda_graph: bool = True,
) -> None:
    """Lock the Qwen thinking / NVIDIA cookbook knobs shared by both GPQA recipes."""
    assert data["model"] == LOCAL_EXPORT
    assert data["model"] != HUB_BF16
    assert data["thinking"]["reasoning_effort"] == "xhigh"
    assert data["sampling"]["temperature"] == 0.0
    assert data["sampling"]["top_p"] == 0.95
    assert data["sampling"]["top_k"] == 20
    assert data["generation"]["max_new_tokens"] == 0
    assert data["generation"]["seed"] == 0
    assert data["generation"]["max_model_len"] == 262144
    serve = data["serve"]
    assert serve["engine"] == "sglang"
    assert serve["attention_backend"] == attention_backend
    assert serve["kv_offloading_size_gb"] == kv_offloading_size_gb
    assert serve["disable_cuda_graph"] is disable_cuda_graph


def test_gpqa_eval_recipe_matches_official_cards() -> None:
    data = _load(GPQA)
    _assert_gpqa_official_cards(data, attention_backend="flashinfer")
    assert data["name"] == "gpqa-diamond-qwen38-official"


def test_gpqa_5090_recipe_matches_official_cards_except_triton() -> None:
    data = _load(GPQA_5090)
    _assert_gpqa_official_cards(
        data, attention_backend="triton", kv_offloading_size_gb=64
    )
    assert data["name"] == "gpqa-diamond-qwen38-official-5090"
    assert data.get("concurrency") == 24
    assert data["serve"].get("max_mamba_cache_size") == 96
    assert data["serve"].get("mamba_ssm_dtype") == "bfloat16"
    assert data["serve"].get("max_running_requests") == 24
    serve = data["serve"]
    # Extra 5090 GEMM/sampling knobs may land in later commits; skip if absent.
    optional = {
        "sampling_backend": "pytorch",
        "linear_attn_backend": "triton",
        "fp8_gemm_backend": "triton",
        "fp4_gemm_backend": "marlin",
        "force_fp8_marlin": True,
    }
    for key, value in optional.items():
        if key in serve:
            assert serve[key] == value, key


def test_gpqa_6000d_recipe_matches_official_cards_with_graphs() -> None:
    data = _load(GPQA_6000D)
    _assert_gpqa_official_cards(
        data,
        attention_backend="triton",
        kv_offloading_size_gb=0,
        disable_cuda_graph=False,
    )
    assert data["name"] == "gpqa-diamond-qwen38-6000d"
    assert data["concurrency"] == 64
    serve = data["serve"]
    assert serve["max_running_requests"] == 64
    assert serve["max_mamba_cache_size"] == 256
    assert serve["mamba_ssm_dtype"] == "bfloat16"
    assert serve["mem_fraction_static"] == 0.90
    assert serve["chunked_prefill_size"] == 4096
    assert serve["fp8_gemm_backend"] == "cutlass"
    assert serve["fp4_gemm_backend"] == "marlin"
    assert serve["enable_hierarchical_cache"] is False
    assert serve["disable_radix_cache"] is True
    assert serve["disable_silu_fp4_quant_fusion"] is True
    assert serve["flashinfer_available"] is False


def test_gpqa_5090_recipe_stays_distinct_from_default() -> None:
    default = _load(GPQA)
    fivek = _load(GPQA_5090)
    assert default["name"] != fivek["name"]
    assert default["serve"]["attention_backend"] == "flashinfer"
    assert fivek["serve"]["attention_backend"] == "triton"
    assert default["serve"]["attention_backend"] != fivek["serve"]["attention_backend"]
    assert default["serve"]["engine"] == fivek["serve"]["engine"] == "sglang"
    assert GPQA.read_text() != GPQA_5090.read_text()

