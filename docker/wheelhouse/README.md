# Spark Offline Wheels

This directory is empty in git. Both Spark Dockerfiles search it before using
PyPI; `--build-arg PIP_NO_INDEX=1 --network=none` disables package network access.
The pinned official base image must already be available locally.

Download the exact ARM64 wheels required by the selected base into this
directory with `python -m pip download --no-deps --only-binary=:all:`. Use
`docker/requirements-build-spark.txt` for both images. The PTQ NGC base needs
these additional fixed versions:

```text
accelerate==1.14.0
annotated-doc==0.0.5
nvidia-modelopt==0.47.0
sentencepiece==0.2.1
shellingham==1.5.4
tokenizers==0.22.2
transformers==5.12.1
typer==0.27.2
```

Use Python 3.12 / Linux ARM64 for download resolution. When downloading from a
different platform, pass the ARM64 platform, Python and ABI options to pip.
Check each wheel's SHA256 against its official PyPI release metadata and keep
the resulting `SHA256SUMS` with the archived wheels. The complete runtime pins
in `docker/requirements-gpu-spark.txt` also assert the packages inherited from
the immutable NGC base, including its native CUDA Torch build.

```bash
docker build --platform linux/arm64 --no-cache --network=none \
  --build-arg PIP_NO_INDEX=1 --build-arg GIT_REVISION="$(git rev-parse HEAD)" \
  -f Dockerfile.spark -t megaquant:gb10 .
```

Use the same options with `Dockerfile.sglang.spark` for serving. Wheel binaries
and checksum files are ignored by git; include them in the build context or an
offline archive separately.
