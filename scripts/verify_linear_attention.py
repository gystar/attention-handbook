#!/usr/bin/env python3
"""Finite-feature kernel identities and causal states; no model or GPU benchmark."""
import json, math, random, platform

def read(q,s,z):
    denom=sum(a*b for a,b in zip(q,z))
    if denom<=0:raise ValueError('positive denominator required')
    return [sum(q[a]*s[a][b] for a in range(len(q)))/denom for b in range(len(s[0]))]

def recurrent(q,k,v,split=None):
    d,dv=len(k[0]),len(v[0]);s=[[0.]*dv for _ in range(d)];z=[0.]*d;out=[]
    for i in range(len(k)):
        if split is not None and i==split:
            # State serialization at a chunk boundary, without changing values.
            s=json.loads(json.dumps(s));z=json.loads(json.dumps(z))
        for a in range(d):
            z[a]+=k[i][a]
            for b in range(dv):s[a][b]+=k[i][a]*v[i][b]
        out.append(read(q[i],s,z))
    return out

def main():
    rng=random.Random(20261026);err=0.;cases=0
    for n in (1,2,7,19):
      for d in (1,3,8):
       for dv in (1,4):
        for _ in range(5):
         q=[[math.exp(rng.uniform(-2,2)) for _ in range(d)] for _ in range(n)]
         k=[[math.exp(rng.uniform(-2,2)) for _ in range(d)] for _ in range(n)]
         v=[[rng.uniform(-2,2) for _ in range(dv)] for _ in range(n)]
         out=recurrent(q,k,v)
         assert out==recurrent(q,k,v,n//2)
         for t in range(n):
          scores=[sum(a*b for a,b in zip(q[t],key)) for key in k[:t+1]]
          expected=[sum(scores[j]*v[j][b] for j in range(t+1))/sum(scores) for b in range(dv)]
          err=max(err,*(abs(a-b) for a,b in zip(out[t],expected)))
         cases+=1
    assert err<1e-12
    toy=recurrent([[1,1]]*3,[[1,0],[0,1],[1,1]],[[2,0],[0,4],[3,3]])
    assert toy==[[2.,0.],[1.,2.],[2.,2.5]]
    try:read([0],[[1]],[1])
    except ValueError:pass
    else:raise AssertionError('zero denominator')
    print(json.dumps(dict(python=platform.python_version(),scope='fixed features; causal explicit sums versus recurrence; chunk state continuity; no training or hardware evaluation',cases=cases,max_error=err,toy_outputs=toy,softmax_example=math.e/(1+math.e),elu_kernel_example=2/3,zero_denominator_rejected=True,status='passed'),indent=2))

if __name__=='__main__':main()
