#!/usr/bin/env python3
"""Connection counts, structural reachability and a one-layer pruning bound."""
import json,math,random,platform


def main():
    cases=0
    for n in range(1,25):
        for w in range(1,n+1):
            rows=[set(range(max(0,i-w+1),i+1)) for i in range(n)]
            assert sum(map(len,rows))==n*w-w*(w-1)//2
            reachable=[{i} for i in range(n)]
            for depth in range(1,6):
                reachable=[set().union(*(reachable[j] for j in row)) for row in rows]
                for i in range(n):assert reachable[i]==set(range(max(0,i-depth*(w-1)),i+1))
            cases+=1
    # A global position gathers in one layer and broadcasts in the next.
    n=8;global_index=0
    rows=[{j for j in range(n) if abs(i-j)<=1 or i==global_index or j==global_index} for i in range(n)]
    assert all(set().union(*(rows[j] for j in row))==set(range(n)) for row in rows)
    causal=[{j for j in row if j<=i} for i,row in enumerate(rows)]
    assert causal[0]=={0}
    active={(i//4,i//4) for i in range(8)}
    assert len(active)*4*4==32
    p=[.2,.3,.5];v=[1,2,10]
    original=sum(a*b for a,b in zip(p,v));sparse=sum(a*b for a,b in zip(p[:2],v[:2]))/sum(p[:2])
    assert math.isclose(original,5.8) and math.isclose(sparse,1.6)
    rng=random.Random(20261026);max_identity_error=0
    for n in [3,7,13]:
        for _ in range(100):
            logits=[rng.uniform(-8,8) for _ in range(n)];m=max(logits)
            weights=[math.exp(x-m) for x in logits];total=sum(weights);weights=[x/total for x in weights]
            values=[[rng.uniform(-2,2) for _ in range(3)] for _ in range(n)]
            keep=set(rng.sample(range(n),rng.randrange(1,n)))
            delta=sum(weights[j] for j in range(n) if j not in keep)
            kept=sum(weights[j] for j in keep)
            a=[sum(weights[j]*values[j][d] for j in keep)/kept for d in range(3)]
            r=[sum(weights[j]*values[j][d] for j in range(n) if j not in keep)/delta for d in range(3)]
            out=[sum(weights[j]*values[j][d] for j in range(n)) for d in range(3)]
            err=max(abs((out[d]-a[d])-delta*(r[d]-a[d])) for d in range(3))
            max_identity_error=max(max_identity_error,err)
            assert err<1e-12
            norm=lambda x:math.sqrt(sum(y*y for y in x))
            B=max(map(norm,values));actual=norm([x-y for x,y in zip(out,a)])
            assert actual<=2*delta*B+1e-12
    # Page-wise extrema bound every dot product, including negative query coordinates.
    worst_slack=0
    for _ in range(300):
        query=[rng.uniform(-3,3) for _ in range(4)]
        page=[[rng.uniform(-2,2) for _ in range(4)] for _ in range(7)]
        bound=sum(max(query[a]*min(k[a] for k in page),query[a]*max(k[a] for k in page)) for a in range(4))
        exact=max(sum(x*y for x,y in zip(query,k)) for k in page)
        assert bound+1e-12>=exact
        worst_slack=max(worst_slack,bound-exact)
    query=[1,1];pages=[[[1,0],[0,1]],[[.75,.75],[.75,.75]]]
    bounds=[sum(max(k[a] for k in page) for a in range(2)) for page in pages]
    maxima=[max(sum(k) for k in page) for page in pages]
    assert bounds==[2,1.5] and maxima==[1,1.5]
    # Distinguish independent branch normalization from a single union Softmax.
    # Each branch has a single value and a sigmoid gate of 0.5.
    gated=.5*0+.5*10
    union=10*math.exp(2)/(1+math.exp(2))
    assert abs(gated-union)>3
    print(json.dumps({'python':platform.python_version(),'scope':'fixed-pattern counts, structural reachability, fixed-input pruning algebra; no model-quality or GPU measurements','window_configurations':cases,'depths_checked':[1,2,3,4,5],'random_pruning_examples':300,'max_identity_error':max_identity_error,'scalar_example':{'dense':original,'sparse':sparse},'block_example':{'logical_pairs':8,'scheduled_slots':32},'page_bound_tests':300,'page_bound_example':{'bounds':bounds,'actual_maxima':maxima},'branch_example':{'gated':gated,'union_softmax':union},'status':'passed'},indent=2))

if __name__=='__main__':main()
