"""Pure local packaging and bootstrap planning for matched PTQ R6.

This module only reads local inputs and returns a deterministic package plus a
container launch plan.  It never creates directories, invokes Docker, opens a
network connection, or uploads a payload.  The returned plan is deliberately
execution-disabled and is intended for a separately reviewed target-side
installer.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import tarfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = next(
    parent
    for parent in (HERE, *HERE.parents)
    if all((parent / marker).exists() for marker in ("pyproject.toml", "src", "recipes"))
)
DEFAULT_RECIPE = ROOT / "recipes/qwen3.8-27b-nvfp4-w4a4.spark.yaml"
OWNER_PREFIX = "megaquant-matched-r6-"
OWNER = "matched-ptq-r6-1022"
OWNER_LABEL = "io.megaquant.matched-r6.owner"
JOB_LABEL = "io.megaquant.matched-r6.job"
IMAGE_REF = "megaquant@sha256:d3f9770ae061eeedd10100971ce77e2ac253cf5a45914b3d65c1bee86847dfe1"
IMAGE_ID = "sha256:d3f9770ae061eeedd10100971ce77e2ac253cf5a45914b3d65c1bee86847dfe1"
IMAGE_REF_RE = re.compile(r"^[^\s@]+(?:/[^\s@]+)*@sha256:[0-9a-f]{64}$")
IMAGE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
JOB_RE = re.compile(r"^[a-z0-9]{16,32}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ARM_TIMEOUT_SECONDS = 4 * 60 * 60
PAIR_TIMEOUT_SECONDS = 2 * ARM_TIMEOUT_SECONDS
EXPERIMENT_TTL_SECONDS = 2 * ARM_TIMEOUT_SECONDS + 15 * 60
WORKER_TTL_SECONDS = EXPERIMENT_TTL_SECONDS + 15 * 60


class BootstrapError(ValueError):
    """An input cannot be represented by the fail-closed bootstrap plan."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    """Hash a file or directory, including stable relative names for trees."""
    path = Path(path)
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise BootstrapError(f"input path does not exist: {path}")
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        relative = item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _manifest_digest(manifest: Mapping[str, Mapping[str, Any]]) -> str:
    return sha256_bytes(_canonical_json(manifest))


def _validate_hash(value: str, field: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise BootstrapError(f"{field} must be a lowercase SHA-256 digest")
    return value


def rewrite_recipe(recipe: Path | str = DEFAULT_RECIPE) -> bytes:
    """Rewrite only the three staged paths while retaining all policy flags.

    The three path values are rewritten within their named YAML sections.  The
    rest of the source text is retained byte-for-byte, so calibration sample
    count, sequence length, batch size, seed, image policy, and vision/MTP
    policy cannot be silently dropped by a serializer.
    """
    source = Path(recipe)
    try:
        lines = source.read_text(encoding="utf-8").splitlines(keepends=True)
    except OSError as exc:
        raise BootstrapError("could not read recipe") from exc
    replacements = {
        ("model", "source"): "/model",
        ("calibration", "dataset"): "/calibration/calibration.jsonl",
        ("export", "output_dir"): "/output/export",
    }
    seen: set[tuple[str, str]] = set()
    section: str | None = None
    rendered: list[str] = []
    for line in lines:
        stripped = line.lstrip()
        heading = stripped.rstrip("\r\n")
        if (
            heading
            and not heading.startswith("#")
            and not line[0].isspace()
            and heading.endswith(":")
        ):
            section = heading[:-1].strip()
        match = re.match(r"^(\s{2})(source|dataset|output_dir):(\s*)(.*?)(\r?\n)?$", line)
        if match and section is not None:
            key = (section, match.group(2))
            if key in replacements:
                newline = match.group(5) or ""
                line = f"{match.group(1)}{match.group(2)}: {replacements[key]}{newline}"
                seen.add(key)
        rendered.append(line)
    if set(replacements) != seen:
        missing = ", ".join(f"{section}.{key}" for section, key in sorted(set(replacements) - seen))
        raise BootstrapError(f"recipe is missing required path fields: {missing}")
    return "".join(rendered).encode("utf-8")


def _tree_files(root: Path, prefix: str) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for item in sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    ):
        files[f"{prefix}/{item.relative_to(root).as_posix()}"] = item.read_bytes()
    return files


def _package_files(recipe_path: Path) -> dict[str, bytes]:
    required = {
        "runner.py": HERE / "runner.py",
        "pair_compare.py": HERE / "pair_compare.py",
        "pair_supervisor.py": HERE / "pair_supervisor.py",
        "pyproject.toml": ROOT / "pyproject.toml",
    }
    files: dict[str, bytes] = {}
    for name, path in required.items():
        if not path.is_file():
            raise BootstrapError(f"missing package input: {path}")
        files[name] = path.read_bytes()
    files.update(_tree_files(ROOT / "src/megaquant", "src/megaquant"))
    # Keep the recipes root explicit even though only one reviewed recipe is staged.
    files["recipes/.r6-package-marker"] = b"megaquant-matched-ptq-r6-1022\n"
    files["recipe.yaml"] = rewrite_recipe(recipe_path)
    return dict(sorted(files.items()))


def _deterministic_tar(files: Mapping[str, bytes]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        for name in sorted(files):
            if not name or name.startswith("/") or ".." in Path(name).parts:
                raise BootstrapError("unsafe package path")
            data = files[name]
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o444
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            archive.addfile(info, io.BytesIO(data))
    return output.getvalue()


def build_package(recipe: Path | str = DEFAULT_RECIPE) -> dict[str, Any]:
    """Return deterministic archive bytes and a JSON-safe file manifest."""
    recipe_path = Path(recipe)
    files = _package_files(recipe_path)
    manifest = {
        name: {"sha256": sha256_bytes(data), "size": len(data)} for name, data in files.items()
    }
    archive = _deterministic_tar(files)
    return {
        "schema": "matched-ptq-r6-1022-package-v1",
        "manifest": manifest,
        "manifest_sha256": _manifest_digest(manifest),
        "archive_sha256": sha256_bytes(archive),
        "archive_bytes": archive,
    }


def _index_path(model: Path, explicit: Path | str | None) -> Path:
    if explicit is not None:
        value = Path(explicit)
        if not value.is_file():
            raise BootstrapError(f"model index does not exist: {value}")
        return value
    if model.is_file() and model.name.endswith(".index.json"):
        return model
    candidates = sorted(model.glob("*.index.json")) if model.is_dir() else []
    if len(candidates) != 1:
        raise BootstrapError("model index must be supplied or resolve to exactly one *.index.json")
    return candidates[0]


def _calibration_file(calibration: Path) -> Path:
    if calibration.is_file():
        if calibration.name != "calibration.jsonl":
            raise BootstrapError("calibration file must be named calibration.jsonl")
        return calibration
    target = calibration / "calibration.jsonl"
    if not target.is_file():
        raise BootstrapError(f"missing calibration file: {target}")
    return target


def _mount(source: Path, destination: str, readonly: bool) -> str:
    mode = ",readonly" if readonly else ""
    return f"type=bind,src={source},dst={destination}{mode}"


def build_plan(
    *,
    model: Path | str,
    calibration: Path | str,
    output: Path | str,
    evidence: Path | str,
    image_ref: str = IMAGE_REF,
    image_id: str = IMAGE_ID,
    recipe: Path | str = DEFAULT_RECIPE,
    model_index: Path | str | None = None,
    job: str | None = None,
    staging_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build a deterministic, execution-disabled root-owned staging plan."""
    if not IMAGE_REF_RE.fullmatch(image_ref):
        raise BootstrapError("image_ref must be an immutable @sha256 reference")
    if not IMAGE_ID_RE.fullmatch(image_id):
        raise BootstrapError("image_id must be a concrete sha256 image ID")
    if image_ref != IMAGE_REF or image_id != IMAGE_ID:
        raise BootstrapError("image identity must match the read-only R6 image attestation")
    model_path, calibration_path = Path(model), Path(calibration)
    if not model_path.exists() or not model_path.is_dir():
        raise BootstrapError("model must be an existing directory")
    calibration_file = _calibration_file(calibration_path)
    index = _index_path(model_path, model_index)
    package = build_package(recipe)
    source_sha = package["manifest_sha256"]
    model_sha = sha256_tree(model_path)
    index_sha = sha256_file(index)
    calibration_sha = sha256_file(calibration_file)
    output_path, evidence_path = Path(output), Path(evidence)
    if not output_path.is_absolute() or not evidence_path.is_absolute():
        raise BootstrapError("output and evidence paths must be absolute")
    calibration_source = calibration_path if calibration_path.is_dir() else calibration_file.parent
    identity = f"{source_sha}:{model_sha}:{index_sha}:{calibration_sha}:{image_ref}:{image_id}"
    derived_job = sha256_bytes(identity.encode())[:16]
    job = derived_job if job is None else job
    if not JOB_RE.fullmatch(job):
        raise BootstrapError("job must be 16-32 lowercase hexadecimal characters")
    stage = (
        Path(staging_root) if staging_root is not None else Path("/var/lib") / (OWNER_PREFIX + job)
    )
    if not stage.is_absolute() or ".." in stage.parts:
        raise BootstrapError("staging_root must be an absolute path without '..'")
    workspace = stage / "workspace"
    create_argv = [
        "docker",
        "create",
        "--name",
        OWNER_PREFIX + job,
        "--label",
        f"{OWNER_LABEL}={OWNER}",
        "--label",
        f"{JOB_LABEL}={job}",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--gpus",
        "all",
        "--entrypoint",
        "python3",
        "--workdir",
        "/workspace",
        "--env",
        "HF_HUB_OFFLINE=1",
        "--env",
        "TRANSFORMERS_OFFLINE=1",
        "--env",
        "HF_DATASETS_OFFLINE=1",
        "--env",
        "PYTHONNOUSERSITE=1",
        "--env",
        "PYTHONPATH=/workspace/src",
        "--env",
        "XDG_CACHE_HOME=/tmp/cache",
        "--env",
        "TRITON_CACHE_DIR=/tmp/triton",
        "--mount",
        _mount(workspace, "/workspace", True),
        "--mount",
        _mount(model_path.resolve(), "/model", True),
        "--mount",
        _mount(calibration_source.resolve(), "/calibration", True),
        "--mount",
        _mount(output_path.resolve(), "/output", False),
        "--mount",
        _mount(evidence_path.resolve(), "/evidence", False),
        "--tmpfs",
        "/tmp:rw,exec,nosuid,nodev,size=2g",
        image_ref,
        "/workspace/pair_supervisor.py",
        "--recipe",
        "/workspace/recipe.yaml",
        "--runner",
        "/workspace/runner.py",
        "--evidence-root",
        "/evidence",
        "--compare-script",
        "/workspace/pair_compare.py",
        "--arm-timeout",
        str(ARM_TIMEOUT_SECONDS),
        "--pair-timeout",
        str(PAIR_TIMEOUT_SECONDS),
    ]
    return {
        "schema": "matched-ptq-r6-1022-bootstrap-v1",
        "execution_enabled": False,
        "job": job,
        "image_ref": image_ref,
        "image_id": image_id,
        "expected_sha256": {
            "source": _validate_hash(source_sha, "source_sha256"),
            "model": _validate_hash(model_sha, "model_sha256"),
            "model_index": _validate_hash(index_sha, "model_index_sha256"),
            "calibration": _validate_hash(calibration_sha, "calibration_sha256"),
        },
        "source_sha256": source_sha,
        "model_sha256": model_sha,
        "model_index_sha256": index_sha,
        "calibration_sha256": calibration_sha,
        "inputs": {
            "model": str(model_path.resolve()),
            "model_index": str(index.resolve()),
            "calibration": str(calibration_file.resolve()),
            "recipe": str(Path(recipe).resolve()),
        },
        "staging": {
            "root": str(stage),
            "owner_uid": 0,
            "owner_gid": 0,
            "mode": "0700",
            "workspace": {"path": str(workspace), "mode": "0555", "read_only": True},
            "output": {"path": str(output_path), "mode": "0700", "read_only": False},
            "evidence": {"path": str(evidence_path), "mode": "0700", "read_only": False},
        },
        "mounts": [
            {"source": str(workspace), "destination": "/workspace", "read_only": True},
            {"source": str(model_path.resolve()), "destination": "/model", "read_only": True},
            {
                "source": str(calibration_source.resolve()),
                "destination": "/calibration",
                "read_only": True,
            },
            {"source": str(output_path), "destination": "/output", "read_only": False},
            {"source": str(evidence_path), "destination": "/evidence", "read_only": False},
        ],
        "offline_environment": {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "PYTHONNOUSERSITE": "1",
        },
        "container_argv": create_argv,
        "package": {
            "manifest_sha256": package["manifest_sha256"],
            "archive_sha256": package["archive_sha256"],
            "files": package["manifest"],
        },
        "integration_gaps": [
            (
                "The plan is execution-disabled and does not install files, create root-owned "
                "directories, or invoke Docker."
            ),
            (
                "The read-only remote image attestation does not verify image contents beyond "
                "the pinned ref and concrete image ID."
            ),
        ],
    }


def build_maintenance_plan(bootstrap_plan: Mapping[str, Any]) -> dict[str, Any]:
    """Convert bootstrap output to the exact execution-disabled R6 schema.

    The supervisor command suffix is copied to ``child_argv`` as lifecycle
    metadata. The current Docker adapter starts the configured container
    command and does not separately execute this metadata field.
    """
    if bootstrap_plan.get("execution_enabled") is not False:
        raise BootstrapError("bootstrap plan must remain execution-disabled")
    job = bootstrap_plan.get("job")
    if not isinstance(job, str) or not JOB_RE.fullmatch(job):
        raise BootstrapError("bootstrap plan has an invalid job")
    state_dir = str(Path("/var/lib") / (OWNER_PREFIX + job))
    if bootstrap_plan.get("staging", {}).get("root") != state_dir:
        raise BootstrapError("staging root must match the maintenance-owned state directory")
    argv = bootstrap_plan.get("container_argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise BootstrapError("bootstrap plan is missing container argv")
    try:
        image_position = argv.index(bootstrap_plan.get("image_ref"))
    except ValueError as exc:
        raise BootstrapError("container argv does not contain its pinned image") from exc
    child_argv = argv[image_position + 1 :]
    if not child_argv or child_argv[0] != "/workspace/pair_supervisor.py":
        raise BootstrapError("container argv has an unexpected supervisor command")
    result = {
        "job": job,
        "state_dir": state_dir,
        "image_ref": bootstrap_plan.get("image_ref"),
        "image_id": bootstrap_plan.get("image_id"),
        "owner": OWNER,
        "owner_label_key": OWNER_LABEL,
        "job_label_key": JOB_LABEL,
        "create_argv": list(argv),
        "child_argv": child_argv,
        "arm_timeout_seconds": ARM_TIMEOUT_SECONDS,
        "experiment_ttl": EXPERIMENT_TTL_SECONDS,
        "worker_ttl": WORKER_TTL_SECONDS,
        "execution_enabled": False,
    }
    if result["image_ref"] != IMAGE_REF or result["image_id"] != IMAGE_ID:
        raise BootstrapError("bootstrap plan image identity does not match the attestation")
    return result


def package_and_plan(**kwargs: Any) -> dict[str, Any]:
    """Convenience wrapper returning both plan metadata and archive bytes."""
    plan = build_plan(**kwargs)
    package = build_package(kwargs.get("recipe", DEFAULT_RECIPE))
    return {"plan": plan, "package": package}
