#!/usr/bin/env python3
"""CPU float64 checks for chapter 6 rowwise streaming attention, not a GPU kernel."""
import json
import math
import platform
import random


def dense(scores, values):
    valid=[i for i,s in enumerate(scores) if s != -math.inf]
    if not valid:
        raise ValueError('all-masked row')
    m=max(scores[i] for i in valid)
    weights=[math.exp(scores[i]-m) for i in valid]
    z=sum(weights)
    return [sum(w*values[i][d] for w,i in zip(weights,valid))/z
            for d in range(len(values[0]))]


def merge(a,b):
    if a is None:return b
    if b is None:return a
    ma,la,ua=a;mb,lb,ub=b
    m=max(ma,mb);alpha=math.exp(ma-m);beta=math.exp(mb-m)
    return m,alpha*la+beta*lb,[alpha*x+beta*y for x,y in zip(ua,ub)]


def state(scores,values,indices):
    valid=[i for i in indices if scores[i] != -math.inf]
    if not valid:return None
    m=max(scores[i] for i in valid)
    w=[math.exp(scores[i]-m) for i in valid]
    return m,sum(w),[sum(a*values[i][d] for a,i in zip(w,valid))
                    for d in range(len(values[0]))]


def streaming(scores,values,blocks):
    acc=None
    for block in blocks:acc=merge(acc,state(scores,values,block))
    if acc is None:raise ValueError('all-masked row')
    _,z,u=acc
    return [x/z for x in u]


def check_backward():
    """Finite differences against a dense forward; backward rebuilds block probabilities."""
    rng=random.Random(20261027)
    errors={};count=0
    nq,nk,dh,dv=3,4,2,3
    def matrix(n,d):return [[rng.uniform(-.8,.8) for _ in range(d)] for _ in range(n)]
    q,k,v,g=matrix(nq,dh),matrix(nk,dh),matrix(nk,dv),matrix(nq,dv)
    c=1/math.sqrt(dh)
    def forward(mask,z):
        outputs=[];logs=[]
        for i in range(nq):
            s=[c*sum(x*y for x,y in zip(q[i],k[j])) if mask[i][j] else -math.inf for j in range(nk)]
            m=max(s);w=[math.exp(x-m) for x in s];total=sum(w)
            logs.append(m+math.log(total))
            outputs.append([sum(w[j]/total*z[i][j]*v[j][a] for j in range(nk)) for a in range(dv)])
        return outputs,logs
    def loss(mask,z):
        out,_=forward(mask,z)
        return sum(out[i][a]*g[i][a] for i in range(nq) for a in range(dv))
    for causal in [False,True]:
        # Rectangular, explicitly left-aligned causal mask; not a library default.
        mask=[[not causal or j<=i for j in range(nk)] for i in range(nq)]
        for dropout in [False,True]:
            z=[[0. if dropout and (i+j)%3==0 else (2. if dropout else 1.) for j in range(nk)] for i in range(nq)]
            out,logs=forward(mask,z)
            D=[sum(x*y for x,y in zip(g[i],out[i])) for i in range(nq)]
            numeric=[];epsilon=1e-6
            for tensor in [q,k,v]:
                grad=[]
                for row in tensor:
                    grow=[]
                    for a in range(len(row)):
                        old=row[a]
                        row[a]=old+epsilon;plus=loss(mask,z)
                        row[a]=old-epsilon;minus=loss(mask,z)
                        row[a]=old
                        grow.append((plus-minus)/(2*epsilon))
                    grad.append(grow)
                numeric.append(grad)
            for block in [1,2,8]:
                dq=[[0.]*dh for _ in range(nq)];dk=[[0.]*dh for _ in range(nk)];dval=[[0.]*dv for _ in range(nk)]
                for startq in range(0,nq,block):
                    for startk in range(0,nk,block):
                        for i in range(startq,min(nq,startq+block)):
                            for j in range(startk,min(nk,startk+block)):
                                if not mask[i][j]:continue
                                score=c*sum(x*y for x,y in zip(q[i],k[j]))
                                p=math.exp(score-logs[i])
                                dp=sum(x*y for x,y in zip(g[i],v[j]))*z[i][j]
                                ds=p*(dp-D[i])
                                for a in range(dv):dval[j][a]+=p*z[i][j]*g[i][a]
                                for a in range(dh):
                                    dq[i][a]+=c*ds*k[j][a]
                                    dk[j][a]+=c*ds*q[i][a]
                for label,actual,expected in zip(['Q','K','V'],[dq,dk,dval],numeric):
                    error=max(abs(x-y) for row,ref in zip(actual,expected) for x,y in zip(row,ref))
                    assert error<1e-8,(causal,dropout,block,label,error)
                    key=f'{label}_causal_{causal}_dropout_{dropout}'
                    errors[key]=max(errors.get(key,0),error)
                    count+=sum(map(len,actual))
    return {'scalar_comparisons':count,'shape':{'Nq':nq,'Nk':nk,'dh':dh,'dv':dv},'block_sizes':[1,2,8],'finite_difference_step':1e-6,'absolute_tolerance':1e-8,'max_errors':errors,'dropout_mask':'fixed deterministic pattern with keep scale 2; tests conditional gradient, not RNG replay','status':'passed'}


def main():
    rng=random.Random(20261026)
    errors={};cases=0
    def check(name,a,b):
        nonlocal cases
        assert all(math.isfinite(x) for x in a+b)
        error=max(abs(x-y) for x,y in zip(a,b))
        assert error<1e-12,(name,error)
        errors[name]=max(errors.get(name,0),error);cases+=1
    scores=[0,math.log(2),math.log(6)]
    values=[[1,0],[0,1],[1,1]]
    check('figure_example',streaming(scores,values,[[0,1],[2]]),[7/9,8/9])
    average=[(a+b)/2 for a,b in zip(dense(scores[:2],values[:2]),values[2])]
    average_error=max(abs(a-b) for a,b in zip(average,[7/9,8/9]))
    assert average_error>.1
    for n in [1,3,7,19]:
        values=[[rng.uniform(-2,2) for _ in range(5)] for _ in range(n)]
        for scale in [1,1000]:
            raw=[rng.uniform(-scale,scale) for _ in range(n)]
            for valid_count in [1,n//2+1,n]:
                scores=[s if i<valid_count else -math.inf for i,s in enumerate(raw)]
                ref=dense(scores,values)
                for size in [1,2,4,32]:
                    blocks=[list(range(i,min(i+size,n))) for i in range(0,n,size)]
                    for order in [blocks,list(reversed(blocks))]:
                        check('streaming_vs_dense',streaming(scores,values,order),ref)
                    # A tree reduction exercises grouping separately from sequential updates.
                    nodes=[state(scores,values,b) for b in blocks]
                    while len(nodes)>1:
                        nodes=[merge(nodes[i],nodes[i+1] if i+1<len(nodes) else None)
                               for i in range(0,len(nodes),2)]
                    _,z,u=nodes[0]
                    check('tree_vs_dense',[x/z for x in u],ref)
    check('large_positive_shift',dense([1000,1000+math.log(2),1000+math.log(6)],[[1,0],[0,1],[1,1]]),[7/9,8/9])
    rejected=0
    for fn in [lambda:dense([-math.inf],[[1]]),lambda:streaming([-math.inf],[[1]],[[0]])]:
        try:fn()
        except ValueError:rejected+=1
    assert rejected==2
    n,dh,element_bytes=8192,128,2
    assert n*n*element_bytes//2**20==128
    assert n*dh*element_bytes//2**20==2
    print(json.dumps({'python':platform.python_version(),'scope':'float64 forward algebra, blockwise backward finite differences, and capacity arithmetic; no low precision or GPU performance validation','seed':20261026,'cases':cases,'absolute_tolerance':1e-12,'max_errors':errors,'unweighted_block_average_error':average_error,'all_masked_rejections':rejected,'backward':check_backward(),'capacity_MiB':{'one_score_matrix':128,'one_input_matrix':2},'status':'passed'},indent=2))

if __name__=='__main__':main()
