# 第 8 章 KV Cache 的存储管理与推理优化

第七章区分了“当前读取哪些历史”和“仍保存哪些历史”。在推理服务中，这些历史数据还需要随着请求进入、增长、暂停和结束而分配与回收。第四章确定每个位置的 KV 表示，本章转向这些数据的生命周期，以及有限显存如何支持持续生成和多个并发请求。

先按实际处理位置计算容量，再介绍分页布局与数值量化；随后讨论历史选择、跨请求复用和数据迁移，最后结合质量、延迟与吞吐量评价缓存策略。各方法改变的对象不同，组合时需要检查数据格式、有效位置和生命周期是否兼容。

- [8.1 KV Cache 的生命周期与容量](lifecycle.md)
- [8.2 PagedAttention 与缓存分配](paged-attention.md)
- [8.3 KV Cache Quantization](quantization.md)
- [8.4 Token Selection 与 Cache Eviction](eviction.md)
- [8.5 Prefix Caching 与 Offloading](reuse-offloading.md)
- [8.6 缓存策略与服务性能](evaluation.md)
