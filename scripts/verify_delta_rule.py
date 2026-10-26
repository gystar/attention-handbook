#!/usr/bin/env python3
"""Delta-rule algebra and local gradient checks; no model training."""
import json,random,math,platform

def read(s,k):return [sum(s[a][b]*k[a] for a in range(len(k))) for b in range(len(s[0]))]
def step(s,k,v,beta,alpha=1.):
 old=read(s,k)
 return [[alpha*s[a][b]+beta*k[a]*(v[b]-alpha*old[b]) for b in range(len(v))] for a in range(len(k))]
def main():
 rng=random.Random(20261026);worst=0.;graderr=0.;nchecks=0
 for dk,dv in [(1,1),(3,2),(2,5)]:
  for _ in range(100):
   s=[[rng.uniform(-1,1) for _ in range(dv)] for _ in range(dk)]
   k=[rng.uniform(-1,1) for _ in range(dk)];v=[rng.uniform(-1,1) for _ in range(dv)]
   beta=rng.random();alpha=rng.random();updated=step(s,k,v,beta,alpha)
   matrix=[[alpha*((1 if a==c else 0)-beta*k[a]*k[c]) for c in range(dk)] for a in range(dk)]
   direct=[[sum(matrix[a][c]*s[c][b] for c in range(dk))+beta*k[a]*v[b] for b in range(dv)] for a in range(dk)]
   worst=max(worst,*(abs(updated[a][b]-direct[a][b]) for a in range(dk) for b in range(dv)))
   norm=math.sqrt(sum(x*x for x in k));unit=[x/norm for x in k]
   replacement=read(step(s,unit,v,1.),unit)
   worst=max(worst,*(abs(a-b) for a,b in zip(replacement,v)))
   old=read(s,k);h=1e-5
   def loss():return sum((a-b)**2 for a,b in zip(read(s,k),v))/2
   for a in range(dk):
    for b in range(dv):
     saved=s[a][b];s[a][b]=saved+h;plus=loss();s[a][b]=saved-h;minus=loss();s[a][b]=saved
     graderr=max(graderr,abs((plus-minus)/(2*h)-k[a]*(old[b]-v[b])));nchecks+=1
 assert worst<1e-12 and graderr<1e-8
 toy=step([[2.],[4.]],[1.,0.],[6.],1.,.5);assert toy==[[6.],[2.]]
 assert step([[2.],[4.]],[1.,0.],[6.],1.)==[[6.],[4.]]
 # Nonunit key disproves unconditional exact replacement.
 assert read(step([[2.]],[2.],[6.],1.),[2.])==[12.]
 print(json.dumps(dict(python=platform.python_version(),cases=300,gradient_components=nchecks,max_algebra_error=worst,max_gradient_error=graderr,gated_toy=toy,nonunit_counterexample=True,scope='local algebra and finite differences; no full model, GPU or low precision',status='passed'),indent=2))
if __name__=='__main__':main()
