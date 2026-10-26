#!/usr/bin/env python3
"""Deterministic scalar checks of Chapter 2 equations; Python stdlib only.
Checks one attention layer, not model quality or hardware performance.
"""
import json
import math
import platform
import random


def matmul(a, b):
    assert len(a[0]) == len(b)
    return [[math.fsum(x*y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def softmax(row):
    maximum = max(row)
    if not math.isfinite(maximum):
        raise ValueError('Each Query needs a finite, allowed Key score')
    exp = [math.exp(x-maximum) for x in row]
    total = math.fsum(exp)
    return [x/total for x in exp]


def attention(q, k, v, allowed=None):
    scores = matmul(q, list(zip(*k)))
    scale = math.sqrt(len(k[0]))
    weights = [softmax([s/scale if allowed is None or allowed[i][j] else -math.inf
                        for j, s in enumerate(row)]) for i, row in enumerate(scores)]
    return matmul(weights, v)


def scalar_reference(q, k, v):
    # Direct weighted sums over source positions, without the matrix routine.
    result = []
    for query in q:
        exp = [math.exp(sum(query[r]*key[r] for r in range(len(query)))/math.sqrt(len(query))) for key in k]
        result.append([sum(exp[j]*v[j][r] for j in range(len(k)))/sum(exp) for r in range(len(v[0]))])
    return result


def mask(nq, nk, offset=0):
    return [[j<=offset+i for j in range(nk)] for i in range(nq)]


def main():
    rng = random.Random(20261026)
    def matrix(n, d):
        return [[rng.uniform(-1,1) for _ in range(d)] for _ in range(n)]
    errors = {}
    def check(name, a, b):
        assert len(a) == len(b) and all(len(x)==len(y) for x,y in zip(a,b)), name
        error = max(abs(x-y) for ra,rb in zip(a,b) for x,y in zip(ra,rb))
        assert error < 1e-12, (name, error)
        errors[name] = error
    q, k, v = [[1]], [[0],[math.log(2)],[math.log(7)]], [[1,0],[0,1],[1,1]]
    check('toy_unmasked', attention(q,k,v), [[.8,.9]])
    check('toy_masked_renormalization', attention(q,k,v,[[True,True,False]]), [[1/3,2/3]])
    check('softmax_shift_invariance', [softmax([1001,1002,1003])], [softmax([1,2,3])])
    try:
        attention(q,k,v,[[False,False,False]])
    except ValueError:
        pass
    else:
        raise AssertionError('All-masked row was not rejected')
    cq, ck, cv = matrix(2,4), matrix(5,4), matrix(5,3)
    check('rectangular_cross_attention', attention(cq,ck,cv), scalar_reference(cq,ck,cv))
    q, k, v = matrix(7,4), matrix(7,4), matrix(7,3)
    full = attention(q,k,v,mask(7,7))
    single = [attention([q[t]],k[:t+1],v[:t+1])[0] for t in range(7)]
    check('full_vs_incremental_cache', full, single)
    chunks=[]
    for start,end in [(0,3),(3,5),(5,7)]:
        chunks.extend(attention(q[start:end],k[:end],v[:end],mask(end-start,end,start)))
    check('full_vs_chunked_cache', full, chunks)
    bad = attention(q[3:5],k[:5],v[:5],mask(2,5))
    bad_error=max(abs(x-y) for a,b in zip(full[3:5],bad) for x,y in zip(a,b))
    assert bad_error>1e-3, 'Example must expose missing cache offset'
    changed_k=k[:3]+matrix(4,4); changed_v=v[:3]+matrix(4,3)
    check('causal_prefix_invariance', full[:3], attention(q,changed_k,changed_v,mask(7,7))[:3])
    h1, h2, wo = matrix(3,2), matrix(3,2), matrix(4,5)
    joined=[a+b for a,b in zip(h1,h2)]
    a,b=matmul(h1,wo[:2]),matmul(h2,wo[2:])
    summed=[[x+y for x,y in zip(ra,rb)] for ra,rb in zip(a,b)]
    check('concat_projection_vs_block_sum', matmul(joined,wo), summed)
    print(json.dumps({'python':platform.python_version(), 'seed':20261026,
        'scope':'single-head equations plus multi-head output projection; no training or GPU benchmark',
        'sequence_length':7,'d_k':4,'d_v':3,'tolerance':1e-12,
        'maximum_absolute_errors':errors,'all_masked_rejected':True,
        'missing_offset_counterexample_error':bad_error,'status':'passed'},indent=2))


if __name__=='__main__':
    main()
