#!/usr/bin/env python3
"""Chapter 4 scalar checks. No model training or hardware measurements."""
import json
import math
import platform
import random
from verify_transformer_attention import attention, matmul, softmax


def main():
    rng = random.Random(20261026)
    def matrix(n, d):
        return [[rng.uniform(-1, 1) for _ in range(d)] for _ in range(n)]
    errors = {}
    for groups in (1, 2, 8):
        queries = [matrix(1, 3) for _ in range(8)]
        keys = [matrix(5, 3) for _ in range(groups)]
        values = [matrix(5, 3) for _ in range(groups)]
        mapping = [a // (8 // groups) for a in range(8)]
        direct = [attention(queries[a], keys[mapping[a]], values[mapping[a]])[0] for a in range(8)]
        # Explicit per-head copies emulate broadcast materialization.
        expanded_k = [[list(row) for row in keys[g]] for g in mapping]
        expanded_v = [[list(row) for row in values[g]] for g in mapping]
        expanded = [attention(queries[a], expanded_k[a], expanded_v[a])[0] for a in range(8)]
        error = max(abs(x-y) for a,b in zip(direct,expanded) for x,y in zip(a,b))
        assert error < 1e-12
        errors[f'grouped_vs_expanded_{groups}'] = error
    x, w0, w1 = matrix(5,4), matrix(4,3), matrix(4,3)
    avg = lambda a,b: [[(x+y)/2 for x,y in zip(ar,br)] for ar,br in zip(a,b)]
    a,b=matmul(x,avg(w0,w1)),avg(matmul(x,w0),matmul(x,w1))
    error=max(abs(x-y) for ar,br in zip(a,b) for x,y in zip(ar,br))
    assert error < 1e-12
    errors['linear_projection_mean'] = error
    first,second,pooled = softmax([0,0])[1],softmax([0,2])[1],softmax([0,1])[1]
    delta=abs((first+second)/2-pooled)
    assert delta>0.04
    assert round(second,4)==0.8808 and round(pooled,4)==0.7311
    assert round((first+second)/2,4)==0.6904
    # Independent direct expanded-K/V evaluation vs latent-space evaluation.
    latent, q, uk, uv, wo = matrix(5,3), matrix(1,2), matrix(2,3), matrix(2,3), matrix(2,4)
    transpose = lambda m: [list(col) for col in zip(*m)]
    expanded_k, expanded_v = matmul(latent,transpose(uk)), matmul(latent,transpose(uv))
    direct = matmul(attention(q,expanded_k,expanded_v),wo)
    qhat = matmul(q,uk)
    scores = matmul(qhat,transpose(latent))[0]
    weights = [softmax([v/math.sqrt(2) for v in scores])]
    z = matmul(weights,latent)
    via_latent = matmul(z,matmul(transpose(uv),wo))
    error=max(abs(x-y) for a,b in zip(direct,via_latent) for x,y in zip(a,b))
    assert error<1e-12
    errors['mla_content_expanded_vs_latent'] = error
    wrong_weights = [softmax([v/math.sqrt(3) for v in scores])]
    wrong = matmul(matmul(wrong_weights,latent),matmul(transpose(uv),wo))
    scale_error=max(abs(x-y) for a,b in zip(direct,wrong) for x,y in zip(a,b))
    assert scale_error>1e-5
    qr, kr = matrix(1,2), matrix(5,2)  # Already rotated position vectors.
    combined_q = [q[0]+qr[0]]
    combined_k = [a+b for a,b in zip(expanded_k,kr)]
    direct_rope = matmul(attention(combined_q,combined_k,expanded_v),wo)
    position_scores = matmul(qr,transpose(kr))[0]
    weights_rope = [softmax([(a+b)/math.sqrt(4) for a,b in zip(scores,position_scores)])]
    latent_rope = matmul(matmul(weights_rope,latent),matmul(transpose(uv),wo))
    error=max(abs(x-y) for a,b in zip(direct_rope,latent_rope) for x,y in zip(a,b))
    assert error<1e-12
    errors['mla_decoupled_rope_expanded_vs_latent'] = error
    capacities = {'MLA':512+64,'GQA-2':2*2*128,'GQA-4':2*4*128}
    assert capacities == {'MLA':576,'GQA-2':512,'GQA-4':1024}
    assert capacities['GQA-2'] < capacities['MLA'] < capacities['GQA-4']
    print(json.dumps({'python':platform.python_version(),'seed':20261026,
        'scope':'head grouping, mean projection, MLA content absorption, counterexamples; no model training or performance evaluation',
        'tolerance':1e-12,'errors':errors,'cache_elements_per_layer_per_position':capacities,'wrong_latent_scale_error':scale_error,
        'pooling_example':{'heads':[first,second],'mean_head_output':(first+second)/2,'pooled_key_output':pooled,'difference':delta},
        'status':'passed'},indent=2))


if __name__=='__main__':
    main()
