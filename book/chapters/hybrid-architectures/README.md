# 第 10 章 Hybrid Architectures 的结构设计与方法比较

显式 KV 允许 Query 直接比较历史条目，固定维度状态则将历史汇总后再读取。前者的存储随保留条目增长，后者需要在有限状态中组织信息。Hybrid Architectures 将不同序列模块放入同一模型，让局部比较、较远历史读取和递推更新通过网络层次共同参与建模。

本章按组合对象展开：先看不同读取范围的 Attention，再看 Attention 与 Linear Attention、State Space Models 的组合。Gemma 2、Griffin、Jamba、Samba 等模型用于说明具体布局及其证据，层比例与窗口均随案例标明。之后汇总各层状态，并单独讨论 FFN 侧的 MoE 路由，最后比较完整模型的质量与资源成本。

- [10.1 序列模块的组合方式](composition.md)
- [10.2 Full Attention 与 Sparse Attention 的组合](full-sparse.md)
- [10.3 Attention 与 Linear Attention 的组合](attention-linear.md)
- [10.4 Attention 与 State Space Models 的组合](attention-ssm.md)
- [10.5 层间配置与状态管理](state-management.md)
- [10.6 MoE 与 Expert Routing](moe-routing.md)
- [10.7 模型能力与资源成本的比较](evaluation.md)
