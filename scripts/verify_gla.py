#!/usr/bin/env python3
"""Check row-gated recurrence against its history expansion in float64."""
import json,math,platform,random

def main():
 rng=random.Random(20261026);worst=0.;cases=0
 for n in [1,3,17]:
  for dk,dv in [(1,1),(3,2),(2,5)]:
   for _ in range(20):
    k=[[rng.uniform(-2,2) for _ in range(dk)] for _ in range(n)]
    v=[[rng.uniform(-2,2) for _ in range(dv)] for _ in range(n)]
    gate=[[rng.uniform(.01,1) for _ in range(dk)] for _ in range(n)]
    state=[[0.]*dv for _ in range(dk)]
    for t in range(n):
     for a in range(dk):
      for b in range(dv):state[a][b]=gate[t][a]*state[a][b]+k[t][a]*v[t][b]
     expanded=[[sum(math.prod(gate[r][a] for r in range(j+1,t+1))*k[j][a]*v[j][b] for j in range(t+1)) for b in range(dv)] for a in range(dk)]
     worst=max(worst,*(abs(state[a][b]-expanded[a][b]) for a in range(dk) for b in range(dv)))
    cases+=1
 assert worst<1e-12
 values=[[g**lag for lag in [1,4,16]] for g in [.5,.9]]
 assert .5*4+1==3 and .5*(4+1)==2.5
 print(json.dumps(dict(python=platform.python_version(),scope='row-gated state expansion only; no full GLA model, gradients, low precision or GPU evaluation',cases=cases,max_error=worst,decay_table=values,state_gate_example=3,output_gate_example=2.5,status='passed'),indent=2))
if __name__=='__main__':main()
