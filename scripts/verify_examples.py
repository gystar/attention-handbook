#!/usr/bin/env python3
"""首章教学算例：容量运算和两种 Causal Linear Attention 计算的数值对照。"""
import json
import math


def main():
    weights, values = [0.1, 0.2, 0.7], [(1, 0), (0, 1), (1, 1)]
    output = [sum(w * v[d] for w, v in zip(weights, values)) for d in range(2)]
    assert all(math.isclose(x, y) for x, y in zip(output, [0.8, 0.9]))
    capacities = {name: 2 * 32 * 1 * 8192 * heads * 128 * 2 for name, heads in [("MHA", 32), ("GQA", 8), ("MQA", 1)]}
    assert capacities == {"MHA": 4 * 2**30, "GQA": 2**30, "MQA": 128 * 2**20}
    assert sum(range(1, 6)) == 5 * 6 // 2 == 15

    # 固定非负特征映射 ELU(x)+1；对比显式历史求和与递推状态读取。
    def phi(xs):
        return [x + 1 if x >= 0 else math.exp(x) for x in xs]

    queries = [(0.2, -0.4), (1.0, 0.5), (-0.6, 0.3), (0.8, -0.1)]
    keys = [(0.5, -0.1), (-0.2, 0.7), (0.4, 0.9), (-0.3, 0.2)]
    values = [(1.0, 0.0, 0.5), (0.0, 2.0, -1.0), (1.0, 1.0, 0.0), (-0.5, 0.3, 1.0)]
    state, normalizer = [[0.0] * 3 for _ in range(2)], [0.0] * 2
    max_error = 0.0
    for t, (query, key, value) in enumerate(zip(queries, keys, values)):
        pq, pk = phi(query), phi(key)
        for a in range(2):
            normalizer[a] += pk[a]
            for d in range(3):
                state[a][d] += pk[a] * value[d]
        denominator = sum(pq[a] * normalizer[a] for a in range(2))
        recurrent = [sum(pq[a] * state[a][d] for a in range(2)) / denominator for d in range(3)]
        scores = [sum(a * b for a, b in zip(pq, phi(k))) for k in keys[:t + 1]]
        explicit = [sum(score * v[d] for score, v in zip(scores, values[:t + 1])) / sum(scores) for d in range(3)]
        max_error = max(max_error, *(abs(a - b) for a, b in zip(recurrent, explicit)))
    assert max_error < 1e-12
    print(json.dumps({"weighted_sum": output, "kv_bytes": capacities, "causal_pairs_N5": 15, "linear_attention_steps": 4, "linear_attention_max_abs_error": max_error, "result": "passed"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
