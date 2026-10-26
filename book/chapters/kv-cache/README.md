# 第 8 章 KV Cache 的存储管理与推理优化

KV Cache 随上下文增长，也随并发请求的进入、暂停和结束不断变化。第四章讨论模型需要保存哪些表示；本章讨论服务系统如何分配、压缩、保留和复用这些数据。

先按实际处理位置计算容量，再介绍分页布局与数值量化；随后讨论历史选择、跨请求复用和数据迁移，最后结合质量、延迟与吞吐量评价缓存策略。各方法改变的对象不同，组合时需要检查数据格式、有效位置和生命周期是否兼容。

- [8.1 KV Cache 的生命周期与容量](lifecycle.md)
- [8.2 PagedAttention 与缓存分配](paged-attention.md)
- [8.3 KV Cache Quantization](quantization.md)
- [8.4 Token Selection 与 Cache Eviction](eviction.md)
- [8.5 Prefix Caching 与 Offloading](reuse-offloading.md)
- [8.6 缓存策略与服务性能](evaluation.md)
