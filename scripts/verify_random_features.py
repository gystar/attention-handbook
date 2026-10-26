#!/usr/bin/env python3
"""Deterministic algebra checks, not Monte Carlo accuracy or FAVOR+ benchmarks."""
import json,math,random

def main():
 rng=random.Random(20261026);gaussian_error=0.;ratio_error=0.;reorder_error=0.
 for _ in range(200):
  dk=3;dv=2;m=7;nkeys=5
  vec=lambda d:[rng.uniform(-.5,.5) for _ in range(d)]
  q=vec(dk);keys=[vec(dk) for _ in range(nkeys)];values=[vec(dv) for _ in keys]
  dot=lambda a,b:sum(x*y for x,y in zip(a,b))
  x=[a/dk**.25 for a in q];ys=[[a/dk**.25 for a in k] for k in keys]
  for y,k in zip(ys,keys):
   via_mgf=math.exp((sum((a+b)**2 for a,b in zip(x,y))-dot(x,x)-dot(y,y))/2)
   exact=math.exp(dot(q,k)/math.sqrt(dk));gaussian_error=max(gaussian_error,abs(via_mgf-exact))
  omega=[[rng.gauss(0,1) for _ in range(dk)] for _ in range(m)]
  feature=lambda a:[math.exp(dot(w,a)-dot(a,a)/2)/math.sqrt(m) for w in omega]
  fq=feature(x);fk=[feature(y) for y in ys]
  estimated=[dot(fq,k) for k in fk];exact=[math.exp(dot(x,y)) for y in ys]
  d=sum(exact);dh=sum(estimated)
  num=[sum(w*v[b] for w,v in zip(exact,values)) for b in range(dv)]
  nh=[sum(w*v[b] for w,v in zip(estimated,values)) for b in range(dv)]
  state=[[sum(k[a]*v[b] for k,v in zip(fk,values)) for b in range(dv)] for a in range(m)]
  z=[sum(k[a] for k in fk) for a in range(m)]
  for b in range(dv):
   out=num[b]/d;approx=nh[b]/dh
   ratio_error=max(ratio_error,abs(approx-out-((nh[b]-num[b])-out*(dh-d))/dh))
   reordered=sum(fq[a]*state[a][b] for a in range(m))/dot(fq,z)
   reorder_error=max(reorder_error,abs(reordered-approx))
 assert max(gaussian_error,ratio_error,reorder_error)<1e-12
 print(json.dumps(dict(cases=200,max_mgf_identity_error=gaussian_error,max_ratio_identity_error=ratio_error,max_reorder_error=reorder_error,scope='Gaussian MGF algebra, normalized ratio identity and fixed random-map reorder; no ORF variance or trained-model claims',status='passed'),indent=2))
if __name__=='__main__':main()
