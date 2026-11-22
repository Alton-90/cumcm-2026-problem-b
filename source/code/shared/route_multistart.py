"""Deterministic multistart open-route 2-opt with a distance matrix."""
import math
def route(position,nodes):
 if len(nodes)<2:return list(nodes)
 points=[position]+[x[2] for x in nodes];n=len(nodes)
 d=[[math.dist(a,b) for b in points] for a in points]
 def length(r):return sum(d[a][b] for a,b in zip([0]+r,r))
 best=None;bestcost=float('inf')
 for first in range(1,n+1):
  rem=set(range(1,n+1));rem.remove(first);path=[first]
  while rem:
   j=min(rem,key=lambda j:(d[path[-1]][j],nodes[j-1][0],nodes[j-1][1]));path.append(j);rem.remove(j)
  for _ in range(60):
   gain=1e-8;choice=None
   for i in range(n-1):
    before=0 if i==0 else path[i-1];a=path[i]
    for j in range(i+1,n):
     z=path[j];g=d[before][a]-d[before][z]
     if j+1<n:g+=d[z][path[j+1]]-d[a][path[j+1]]
     if g>gain:gain=g;choice=(i,j)
   if choice is None:break
   i,j=choice;path[i:j+1]=reversed(path[i:j+1])
  cost=length(path)
  if cost<bestcost-1e-7:best,bestcost=path,cost
 return [nodes[i-1] for i in best]
