"""Sparse Walsh rotations with reusable parity and a short closed mask walk.

Binary address encoding is unchanged. No amplitudes, angles, Hamiltonian,
Trotter ordering, or retained coefficient threshold are changed.
"""
from functools import lru_cache
import numpy as np

def walsh(values):
    a=np.asarray(values,dtype=float).copy();size=len(a);h=1
    if size<1 or size&(size-1):raise ValueError('A power-of-two angle table is required.')
    while h<size:
        for start in range(0,size,2*h):
            x=a[start:start+h].copy();y=a[start+h:start+2*h].copy()
            a[start:start+h]=x+y;a[start+h:start+2*h]=x-y
        h*=2
    return a/size

def distance(a,b):return (int(a)^int(b)).bit_count()

def walk_cost(route):
    return sum(distance(a,b) for a,b in zip(route,route[1:]+route[:1]))

def two_opt(route):
    route=list(route);size=len(route)
    while True:
        best=None;gain=0
        for i in range(1,size-1):
            a,b=route[i-1],route[i]
            for j in range(i+1,size):
                c,d=route[j],route[(j+1)%size]
                change=distance(a,c)+distance(b,d)-distance(a,b)-distance(c,d)
                if change<gain:gain=change;best=(i,j)
        if best is None:return route
        i,j=best;route[i:j+1]=route[i:j+1][::-1]

@lru_cache(None)
def mask_walk(active_masks,ncontrols):
    """Choose a closed Hamming walk. Shortest-tour optimality is not assumed."""
    required=set(active_masks)|{0}
    if any(m<0 or m>=2**ncontrols for m in required):raise ValueError('Mask outside control register')
    gray=[i^(i>>1) for i in range(2**ncontrols)]
    if len(required)==2**ncontrols:
        return tuple(gray)  # A closed hypercube Gray cycle.
    candidates=[[m for m in gray if m in required],sorted(required)]
    rng=np.random.default_rng(271027)
    for trial in range(24):
        left=required-{0};route=[0]
        while left:
            nearest=min(distance(route[-1],m) for m in left)
            ties=sorted(m for m in left if distance(route[-1],m)==nearest)
            m=ties[0] if trial==0 else ties[int(rng.integers(len(ties)))]
            route.append(m);left.remove(m)
        candidates.append(route)
    routes=[two_opt(r) for r in candidates]
    route=min(routes,key=lambda r:(walk_cost(r),r))
    assert route[0]==0 and set(route)==required and len(route)==len(required)
    return tuple(route)

def diagonal_terms(terms,controls,target):
    """Implement product exp(-i angle Z_target product(Z_controls)/2)."""
    if target in controls or len(set(controls))!=len(controls):raise ValueError('Invalid register')
    # The caller applies the original coefficient threshold. Do not introduce
    # a second threshold after multiplying by a small time/phase coefficient.
    bymask={int(m):float(a) for m,a in terms}
    if len(bymask)!=len(terms):raise ValueError('Duplicate parity masks')
    if not bymask:return []
    route=mask_walk(tuple(sorted(bymask)),len(controls));current=0;g=[]
    for mask in route:
        changed=current^mask
        g.extend(('cx',[q,target],None) for j,q in enumerate(controls) if (changed>>j)&1)
        if mask in bymask:g.append(('rz',[target],bymask[mask]))
        current=mask
    g.extend(('cx',[q,target],None) for j,q in enumerate(controls) if (current>>j)&1)
    return g

def controlled_ry(angles,controls,target):
    coeff=walsh(angles)
    if len(coeff)!=2**len(controls):raise ValueError('Angle table/control mismatch')
    terms=[(m,a) for m,a in enumerate(coeff) if abs(a)>1e-14]
    return [('sdg',[target],None),('h',[target],None)]+diagonal_terms(terms,controls,target)+[('h',[target],None),('s',[target],None)]

def phase_layer(r,lam,n,probes=(6,7)):
    """Regroup commuting diagonal terms by target; retain every original angle."""
    coeff=walsh(np.asarray(r)**2);groups={}
    for mask,value in enumerate(coeff):
        if abs(value)<1e-14:continue
        inds=[j for j in range(n) if (mask>>j)&1]
        if inds:groups.setdefault(inds[-1],[]).append((inds[:-1],-2*lam*value))
        for probe in probes:groups.setdefault(probe,[]).append((inds,lam*value))
    gates=[]
    for target,terms in sorted(groups.items()):
        controls=sorted({q for inds,_ in terms for q in inds})
        index={q:j for j,q in enumerate(controls)}
        masks=[(sum(1<<index[q] for q in inds),a) for inds,a in terms]
        gates+=diagonal_terms(masks,controls,target)
    return gates
