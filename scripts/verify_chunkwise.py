#!/usr/bin/env python3
"""Compare chunk formulas against token recurrence, including nonzero boundary state."""
import random,json
from verify_delta_rule import step,read

def main():
 rng=random.Random(20261026);worst=0.;cases=0
 for n in (1,3,9):
  for dk,dv in ((1,1),(3,2),(2,4)):
   for _ in range(20):
    vec=lambda d:[rng.uniform(-.5,.5) for _ in range(d)]
    q=[vec(dk) for _ in range(n)];k=[vec(dk) for _ in range(n)];v=[vec(dv) for _ in range(n)]
    initial=[vec(dv) for _ in range(dk)];beta=[rng.random() for _ in range(n)]
    dot=lambda a,b:sum(x*y for x,y in zip(a,b))
    for delta in (False,True):
     s=[row[:] for row in initial];expected=[]
     for i in range(n):
      s=step(s,k[i],v[i],beta[i]) if delta else [[s[a][b]+k[i][a]*v[i][b] for b in range(dv)] for a in range(dk)]
      expected.append(read(s,q[i]))
     r=[]
     for i in range(n):
      old=read(initial,k[i])
      r.append([beta[i]*(v[i][b]-old[b]-sum(dot(k[i],k[j])*r[j][b] for j in range(i))) for b in range(dv)] if delta else v[i])
     out=[[read(initial,q[i])[b]+sum(dot(q[i],k[j])*r[j][b] for j in range(i+1)) for b in range(dv)] for i in range(n)]
     final=[[initial[a][b]+sum(k[i][a]*r[i][b] for i in range(n)) for b in range(dv)] for a in range(dk)]
     worst=max(worst,*(abs(out[i][b]-expected[i][b]) for i in range(n) for b in range(dv)),*(abs(final[a][b]-s[a][b]) for a in range(dk) for b in range(dv)))
     cases+=1
 assert worst<1e-12
 print(json.dumps(dict(cases=cases,max_error=worst,scope='additive and ungated DeltaNet forward chunk output and final state; nonzero entry; no GPU, backward, normalization or Gated DeltaNet',status='passed'),indent=2))
if __name__=='__main__':main()
