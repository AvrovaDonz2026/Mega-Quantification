# 发布记录 / Releases

| 版本 | 说明 | 下载与验证 |
|---|---|---|
| v0.1.3 | [NVFP4 校准与 MTP 设备修复、完整 Spark GPQA 结果](v0.1.3.md) | [GitHub Release](https://github.com/AvrovaDonz2026/Mega-Quantification/releases/tag/v0.1.3) |
| v0.1.2 | [Spark 容器完整重验记录](../validation/dgx-spark-release-20261001.md) | [GitHub Release](https://github.com/AvrovaDonz2026/Mega-Quantification/releases/tag/v0.1.2) |

v0.1.3 附 wheel、源码包、`package-validation.json`、两份聚合 GPU 验证 JSON 和
`SHA256SUMS`。下载附件后可在该目录执行 `sha256sum -c SHA256SUMS`；macOS 使用
`shasum -a 256 -c SHA256SUMS`。

发布包和 GPU 验证各自绑定实际源码 revision。容器构建与运行记录的适用范围见
[验证索引](../validation/README.md)。权重、镜像归档和公共 registry 镜像的发布情况
以具体发布说明为准。

Release assets include package distributions, aggregate validation records and
checksums. See each release note for its exact contents and tested revisions.
