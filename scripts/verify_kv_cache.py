#!/usr/bin/env python3
"""Chapter 8 address/capacity examples and attention invariance; no serving benchmark."""
import json,math,platform,random
from verify_transformer_attention import attention


def main():
    capacity_cases=0
    for block in [1,2,4,16]:
        for length in range(65):
            blocks=(length+block-1)//block
            unused=blocks*block-length
            assert 0<=unused<block
            for t in range(length):assert (t//block)*block+t%block==t
            capacity_cases+=1
    assert (5+7)*128/1024==1.5
    assert (math.ceil(5/4)+math.ceil(7/4))*4*128/1024==2
    rng=random.Random(20261026)
    n,block=9,4;table=[7,2,5]
    k=[[rng.uniform(-1,1) for _ in range(3)] for _ in range(n)]
    v=[[rng.uniform(-1,1) for _ in range(2)] for _ in range(n)]
    q=[[.7,-.2,.5]]
    # Poison unused slots so an incorrect length bound changes the result.
    physical={p:[([100.,100.,100.],[999.,999.]) for _ in range(block)] for p in table}
    for t in range(n):physical[table[t//block]][t%block]=(k[t],v[t])
    pairs=[physical[table[t//block]][t%block] for t in range(n)]
    gathered=attention(q,[x[0] for x in pairs],[x[1] for x in pairs])[0]
    dense=attention(q,k,v)[0]
    error=max(abs(a-b) for a,b in zip(gathered,dense));assert error<1e-12
    reverse=list(reversed(pairs))
    reordered=attention(q,[x[0] for x in reverse],[x[1] for x in reverse])[0]
    reorder_error=max(abs(a-b) for a,b in zip(reordered,dense));assert reorder_error<1e-12
    all_slots=[item for p in table for item in physical[p]]
    wrong=attention(q,[x[0] for x in all_slots],[x[1] for x in all_slots])[0]
    assert max(abs(a-b) for a,b in zip(wrong,dense))>100
    # Copy-on-write of a two-entry partial page, with immutable payloads.
    shared=[('k4','v4'),('k5','v5'),None,None]
    a=list(shared);b=shared
    a[2]=('kA','vA');b[2]=('kB','vB')
    assert a[:2]==b[:2] and a[2]!=b[2] and a is not b
    print(json.dumps({'python':platform.python_version(),'scope':'capacity counts, logical-to-physical gathering, fixed-input attention and copy-on-write illustration; no GPU or allocator implementation validation','capacity_cases':capacity_cases,'mapped_position_6':{'physical_block':table[6//block],'offset':6%block},'gathered_attention_max_error':error,'reordered_attention_max_error':reorder_error,'unused_slot_counterexample_detected':True,'copy_on_write_isolation':True,'status':'passed'},indent=2))

if __name__=='__main__':main()
