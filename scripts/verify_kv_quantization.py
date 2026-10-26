#!/usr/bin/env python3
"""Check Chapter 8 scalar examples; no packed kernel or model-quality validation."""
import json, math, platform, random
from verify_transformer_attention import attention, softmax


def quantize(values, bits):
    lo, hi = min(values), max(values)
    if lo == hi:
        return list(values), 0.
    step = (hi-lo)/(2**bits-1)
    return [lo + step*round((x-lo)/step) for x in values], step


def main():
    rng = random.Random(20261026)
    cases = 0
    max_error = 0.
    for bits in (2, 4, 8):
        for _ in range(100):
            values = [rng.uniform(-20, 30) for _ in range(32)]
            recovered, step = quantize(values, bits)
            assert max(abs(a-b) for a,b in zip(values,recovered)) <= step/2+1e-12
            cases += 1
    assert quantize([3.]*32, 2) == ([3.]*32, 0.)
    for _ in range(100):
        q = [[rng.uniform(-1, 1) for _ in range(4)]]
        k = [[rng.uniform(-2, 2) for _ in range(4)] for _ in range(7)]
        v = [[rng.uniform(-2, 2) for _ in range(3)] for _ in range(7)]
        khat = [list(x) for x in zip(*(quantize(list(c),2)[0] for c in zip(*k)))]
        vhat = [quantize(row,2)[0] for row in v]
        p = softmax([sum(a*b for a,b in zip(q[0],row))/2 for row in k])
        phat = softmax([sum(a*b for a,b in zip(q[0],row))/2 for row in khat])
        o, ohat = attention(q,k,v)[0], attention(q,khat,vhat)[0]
        rhs = [sum((phat[j]-p[j])*v[j][c] + phat[j]*(vhat[j][c]-v[j][c]) for j in range(7)) for c in range(3)]
        err = max(abs(ohat[c]-o[c]-rhs[c]) for c in range(3))
        max_error = max(max_error,err)
    assert max_error < 1e-12
    size = math.ceil(32*2/8) + 2*2
    assert size == 12 and 32*2/size == 64/12
    print(json.dumps(dict(python=platform.python_version(),scope='float64 quantizer bound and fixed-input attention identity; no GPU, bit packing or model evaluation',rounding_bound_cases=cases,constant_group=True,error_identity_cases=100,max_identity_error=max_error,example_bytes=size,example_compression_ratio=64/size,status='passed'),indent=2))


if __name__ == '__main__':
    main()
