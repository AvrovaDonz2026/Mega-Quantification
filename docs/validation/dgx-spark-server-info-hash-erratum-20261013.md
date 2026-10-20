# DGX Spark 配置哈希口径勘误，2026-10-13

**三组完整 server-info 对象均已独立复算，并匹配冻结客户端在生成前记录的
actual 对象摘要。** 先前将 saved 文件字节哈希与 actual 对象哈希不同，解释为
“完整对象身份无法重建”，该限制过强：两字段使用不同的序列化口径。

本报告发布于 2026-10-13，依据的独立 CPU 勘误创建于
`2026-10-12T14:45:23.641451Z`；后续核验时间不倒填为原生成时间。
[相邻 JSON](dgx-spark-server-info-hash-erratum-20261013.json)保留匿名统计与证据哈希。

## 哈希规则与复算结果

冻结客户端的 `saved_server_info_sha256` 对保存文件的原始字节计算 SHA-256；
`actual_server_info_sha256` 对生成请求前新获取的完整 JSON 对象计算摘要，规则为：

```python
hashlib.sha256(
    json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
).hexdigest()
```

其中 `ensure_ascii` 使用 Python 默认的 `True`，`.encode()` 使用默认 UTF-8。
复算处理完整解析对象，包含所有字段，不只比较选定的协议配置。不同编码的哈希
数值不相等，不能单凭这一点判断两个底层对象不同。

| 匿名记录 | saved 原始字节 SHA-256 | 完整对象 canonical JSON SHA-256 |
| --- | --- | --- |
| 题目 3，MSE W4A4 | `ae4741436bddd78973b1da72da43502abadd9c46113e9fb667ee580a1d5600be` | `83c51f9b6b2e3dacc14f6c154f6d91bdf5c42d7b2e9f8be641a1edebef4d9962` |
| 题目 3，max W4A4 | `a6472e80b1e7ecca7ae3f955d68859157c8246f8e5202e29d9779355932a23a1` | `62f33b8143cfe5139527c122dc21707316c7885b503f45d6cfca9246766a2669` |
| 题目 5，源 BF16 | `74b6df65da3fef4386a6ec5458de37e5e36b73fcda836b7360f07683c81fa679` | `92a531f032b38d77a601d3851db89da369c32dc5eb099482a6c8f65b8e112a75` |

三组完整对象的 canonical 摘要均等于对应 recorded actual 摘要；六份 natural /
decoded 记录中的 saved 字节摘要与 actual canonical 摘要全部复算通过，捕获清单
绑定也通过。冻结客户端、driver 和源码清单的哈希一致，源码显示先获取 server-info
并记录对象摘要，再提交生成请求。

## 纠正范围

以下既有报告中，因两种哈希不同而认为完整 actual 对象摘要未被独立重建的限定，
应与本勘误合读：[题目 3 MSE](dgx-spark-repetition-case3-mse-20261012.md)、
[题目 3 max](dgx-spark-repetition-case3-max-20261012.md)及
[题目 5 BF16](dgx-spark-repetition-case5-bf16-20261012.md)。已有日期报告及私有证据
保留原字节；本次只补充完整 canonical 对象与客户端生成前记录摘要的身份绑定。

原始 HTTP 响应的字节格式没有另存，但这不妨碍对象摘要重建。此次本地 CPU 审计
没有发起新服务请求或 GPU 运行，也不独立证明远端进程身份、每个配置字段对应的
现场硬件数值，或生成全过程配置与 KV / SSM 状态不变。种子、容量、并发和数值
路径等既有比较限制仍适用；停止、解析判分、精确循环及语义审阅结论均未改变。

## 证据绑定

| 证据 | SHA-256 |
| --- | --- |
| 独立 CPU 哈希口径勘误 | `4ae2a7147843420e360530a2d5ff4e98d90a92e155721a7017b605e68c0ec7c2` |
| 冻结客户端 | `bd1eca1db4f0d79fc2f8b461ffbf5d84580658686faf5125c9e60ab9a2a90db0` |
| 冻结 driver | `1cc17cecbd3e8c3bc3ae3a9c5dd3ff0ba88221376bcf52238b87eae2687c719e` |
| 冻结源码清单 | `3fb1c96c2f43a5f7b82f3d8a269678c667c0029a0778fe6d892e0988ed4adcae` |

公开记录仅包含匿名统计、规则与哈希，不发布原始配置内容、题目、输出或部署信息。
