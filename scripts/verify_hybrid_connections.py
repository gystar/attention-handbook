#!/usr/bin/env python3
"""Logical connections and graph reachability, not model quality or GPU cost."""
import json

def main():
 counts=0;reachable=0
 for n in range(1,33):
  for w in range(1,17):
   local=sum(len(range(max(0,t-w+1),t+1)) for t in range(n))
   formula=n*w-w*(w-1)//2 if n>=w else n*(n+1)//2
   assert local==formula
   for lf,lw in ((0,3),(1,1),(2,4)):
    direct=sum(t+1 for t in range(n))*lf+local*lw
    assert direct==lf*n*(n+1)//2+lw*formula
    cache=sum(n for _ in range(lf))+sum(min(n,w) for _ in range(lw))
    assert cache==lf*n+lw*min(n,w)
    counts+=1
   deps=[{t} for t in range(n)]
   for depth in range(1,5):
    deps=[set().union(*deps[max(0,t-w+1):t+1]) for t in range(n)]
    assert all(deps[t]==set(range(max(0,t-depth*(w-1)),t+1)) for t in range(n))
    reachable+=1
   # Full layer followed by local layer preserves prefix reachability.
   deps=[set(range(t+1)) for t in range(n)]
   deps=[set().union(*deps[max(0,t-w+1):t+1]) for t in range(n)]
   assert all(deps[t]==set(range(t+1)) for t in range(n))
 rows=[dict(n=n,full=n*(n+1),mixed=n*(n+1)//2+sum(min(t,2) for t in range(1,n+1))) for n in (2,4,8)]
 assert rows==[dict(n=2,full=6,mixed=6),dict(n=4,full=20,mixed=17),dict(n=8,full=72,mixed=51)]
 print(json.dumps(dict(count_cases=counts,local_depth_cases=reachable,full_then_local_cases=512,table_10_3=rows,scope='logical counts, retained KV positions, graph reachability; not learned information fidelity, runtime or allocated memory',status='passed'),indent=2))
if __name__=='__main__':main()
