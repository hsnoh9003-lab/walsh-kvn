"""Wake compiler primitives with a shared one-clean MCX decomposition.

The exact Khattar--Gidney construction is also used by the radial circuit.
Additional work-list entries are unnecessary; the first clean work bit is
restored after each multi-controlled X operation.
"""
from collections import Counter
import numpy as np
def walsh(values):
    a=np.array(values,dtype=float).copy();size=len(a);h=1
    while h<size:
        for start in range(0,size,2*h):
            x=a[start:start+h].copy();y=a[start+h:start+2*h].copy()
            a[start:start+h]=x+y;a[start+h:start+2*h]=x-y
        h*=2
    return a/size


def rz_string(qubits,angle):
    if not qubits:return []
    target=qubits[-1]
    return [('cx',[q,target],None) for q in qubits[:-1]]+[('rz',[target],float(angle))]+[('cx',[q,target],None) for q in reversed(qubits[:-1])]


def controlled_ry(angles,controls,target):
    """Same Walsh angles; reuse parity between commuting rotation terms."""
    from .gray_synthesis import controlled_ry as synthesize
    return synthesize(angles,controls,target)


def ccx(a,b,t):
    # Exact 7-T, 8-Clifford Toffoli. Tdag counts as one magic state.
    return [('h',[t],None),('cx',[b,t],None),('tdg',[t],None),
            ('cx',[a,t],None),('t',[t],None),('cx',[b,t],None),
            ('tdg',[t],None),('cx',[a,t],None),('t',[b],None),
            ('t',[t],None),('h',[t],None),('cx',[a,b],None),
            ('t',[a],None),('tdg',[b],None),('cx',[a,b],None)]


def mcx(controls,target,work):
    m=len(controls)
    if m==0:return [('x',[target],None)]
    if m==1:return [('cx',[controls[0],target],None)]
    if m==2:return ccx(*controls,target)
    if not work:raise ValueError('One clean work qubit is required.')
    from .one_clean_synthesis import clean_mcx, expand
    return expand(clean_mcx(controls,target,work[0]))


def increment(n,work):
    return sum((mcx(list(range(k)),k,work) for k in reversed(range(n))),[])


def inverse(gates):
    inverse_name={'t':'tdg','tdg':'t','s':'sdg','sdg':'s','x':'x','cx':'cx','h':'h'}
    return [(inverse_name.get(name,name),qs,(-angle if angle is not None else None)) for name,qs,angle in reversed(gates)]


def initial_gates(amplitude,n):
    if np.min(amplitude)<-1e-14:raise ValueError('Only nonnegative real input amplitudes are supported.')
    g=[];p=abs(amplitude)**2
    for target in reversed(range(n)):
        width=2**target;angles=[]
        for prefix in range(2**(n-target-1)):
            start=prefix*2*width
            left=p[start:start+width].sum();right=p[start+width:start+2*width].sum()
            angles.append(2*np.arctan2(np.sqrt(right),np.sqrt(left)))
        g+=controlled_ry(angles,list(range(target+1,n)),target)
    return g


def apply_gates(state,gates,nqubits):
    state=np.array(state,dtype=complex,copy=True);basis=np.arange(2**nqubits)
    for name,qs,angle in gates:
        if name=='cx':
            c,t=qs;perm=basis^(((basis>>c)&1)<<t);state=state[perm]
        elif name=='x':state=state[basis^(1<<qs[0])]
        elif name=='h':
            bit=1<<qs[0];lo=basis[(basis&bit)==0];hi=lo+bit
            a=state[lo].copy();b=state[hi].copy()
            state[lo]=(a+b)/np.sqrt(2);state[hi]=(a-b)/np.sqrt(2)
        else:
            bit=(basis>>qs[0])&1
            if name=='rz':phase=np.exp(-.5j*angle*(1-2*bit))
            else:phase=np.exp(1j*{'t':np.pi/4,'tdg':-np.pi/4,'s':np.pi/2,'sdg':-np.pi/2}[name]*bit)
            state*=phase if state.ndim==1 else phase[:,None]
    return state


def inventory(gates):
    c=Counter(g[0] for g in gates)
    return dict(RZ=c['rz'],direct_T=c['t']+c['tdg'],Clifford=sum(v for k,v in c.items() if k not in ['rz','t','tdg']),by_gate=dict(c))


def phase_aligned_error(a,b):
    overlap=np.vdot(b,a)
    return float(np.linalg.norm(a-b*(overlap/abs(overlap))))


