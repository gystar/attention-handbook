# 第 10 章 Hybrid Architectures 的结构设计与方法比较

不同序列模块保存和读取历史的方式不同：Full Attention 保留可逐位置访问的 KV，Local Attention 限定直接读取的窗口，Linear Attention 和 State Space Models 通过递推状态传递历史。Hybrid Architectures 将这些模块放入同一模型，使其共同参与上下文建模。

本章先区分组合发生的位置，再讨论全局与局部 Attention、Attention 与 Linear Attention、Attention 与 State Space Models 的组合，最后分析层间配置、状态管理及比较条件。重点是各模块承担什么计算、保留什么状态，以及组合后付出哪些成本。

- [10.1 序列模块的组合方式](composition.md)
- [10.2 Full Attention 与 Sparse Attention 的组合](full-sparse.md)
- [10.3 Attention 与 Linear Attention 的组合](attention-linear.md)
- [10.4 Attention 与 State Space Models 的组合](attention-ssm.md)
- [10.5 层间配置与状态管理](state-management.md)
- [10.6 模型能力与资源成本的比较](evaluation.md)
