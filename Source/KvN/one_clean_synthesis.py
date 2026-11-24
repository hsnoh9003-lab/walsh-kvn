"""Explicit one-clean MCX and INC/DEC constructions for wake registers.

A: Khattar--Gidney, Quantum 9, 1752 (2025), Sec. 5.1. All compute
   Toffolis are exact CCX here, rather than the optional relative-phase variant.
   The MCX and construction A support arbitrary register widths. Construction B
   below remains the finite six-bit alternative used for radial comparisons.
B: Gidney, Constructing Large Increment Gates (2015), clean split with
   borrowed-bit incrementers. The n=6 finite construction is explicit.

No measurements, resets, extra clean ancillas, or angle-dependent corrections.
Bit lists are least-significant first. Work bits are restored.
"""
from collections import Counter


def cx(c,t): return [('cx',[int(c),int(t)],None)]
def x(t): return [('x',[int(t)],None)]
def tof(a,b,t): return [('ccx',[int(a),int(b),int(t)],None)]


def exact_ccx(a,b,t):
    """Exact seven-T decomposition, matching the previous source compiler."""
    return [('h',[t],None),('cx',[b,t],None),('tdg',[t],None),
            ('cx',[a,t],None),('t',[t],None),('cx',[b,t],None),
            ('tdg',[t],None),('cx',[a,t],None),('t',[b],None),
            ('t',[t],None),('h',[t],None),('cx',[a,b],None),
            ('t',[a],None),('tdg',[b],None),('cx',[a,b],None)]


def inverse(gates):
    names={'t':'tdg','tdg':'t','s':'sdg','sdg':'s','x':'x','cx':'cx','h':'h','ccx':'ccx'}
    return [(names.get(name,name),list(qs),(-angle if angle is not None else None))
            for name,qs,angle in reversed(gates)]


def clean_mcx(controls,target,work):
    """KG conditionally clean ladder, expressed entirely with exact CCX/X."""
    c=list(controls);k=len(c)
    if k==0:return x(target)
    if k==1:return cx(c[0],target)
    if k==2:return tof(*c,target)
    if work in c or target in c or work==target:raise ValueError('Overlapping qubits')
    ladder=[]
    for i in range(1,k-2,2):ladder+=tof(c[i+1],c[i+2],c[i])+x(c[i])
    if (k%2==0 and k>=6) or (k%2 and k>=5):
        a,b,peak=(k-3,k-5,k-6) if k%2==0 else (k-1,k-4,k-5)
        ladder+=tof(c[a],c[b],c[peak])+x(c[peak])
        for i in range(peak,1,-2):ladder+=tof(c[i],c[i-1],c[i-2])+x(c[i-2])
    final=max(0,5-k)
    start=tof(c[0],c[1],work)
    return start+ladder+tof(work,c[final],target)+inverse(ladder)+start


def borrowed_mcx3(controls,target,borrow):
    """Four-CCX toggle-detection C3X; borrowed bit may be arbitrary."""
    a,b,c=controls
    compute=tof(a,b,borrow);action=tof(borrow,c,target)
    return compute+action+compute+action


def borrowed_increment(bits,borrowed):
    """Gidney's many-borrowed increment (used on three bits in B).

    Two reversible subtraction widgets cancel the arbitrary garbage value;
    controlled complements cancel the arbitrary garbage carry. This implements
    INC on bits, identity on all borrowed bits, without a clean input assumption.
    """
    bits=list(bits);borrowed=list(borrowed);n=len(bits)
    if len(borrowed)<n:raise ValueError('One borrowed bit per target bit required')
    borrowed=borrowed[:n]
    if set(bits)&set(borrowed):raise ValueError('Overlapping registers')
    # Cuccaro MAJ/UMA ripple adder is used for the subtraction widget.
    # Unlike Gidney's illustrative Python gist (which expands each rung using
    # two Toffolis), this uses one Toffoli per MAJ/UMA.  The n=3 resulting
    # borrowed increment uses 8 CCX, 22 CX and 5 X.
    sweep_up=[];sweep_down=[]
    for i in range(n-1):
        a,b,c=borrowed[i+1],bits[i],borrowed[i]
        sweep_up+=cx(a,b)+cx(a,c)+tof(b,c,a)
        sweep_down=tof(b,c,a)+cx(a,c)+cx(c,b)+sweep_down
    adder=sweep_up+cx(borrowed[-1],bits[-1])+sweep_down
    widget=inverse(adder)
    carry_toggles=sum((cx(borrowed[0],q) for q in bits[:-1]),[])
    garbage_toggles=sum((x(q) for q in borrowed[1:]),[])
    return x(bits[-1])+carry_toggles+garbage_toggles+widget+garbage_toggles+widget+carry_toggles


def increment_A(bits,work):
    bits=list(bits)
    return sum((clean_mcx(bits[:i],bits[i],work) for i in reversed(range(len(bits)))),[])


def increment_B(bits,work):
    """Six-bit clean split. Three borrowed C3X and two borrowed INC3."""
    bits=list(bits)
    if len(bits)!=6:raise ValueError('This finite B construction is specified for six bits')
    lo,hi=bits[:3],bits[3:]
    carry=borrowed_mcx3(lo,work,hi[0])
    upper_top=borrowed_mcx3([work,hi[0],hi[1]],hi[2],lo[0])
    upper_rest=borrowed_increment([work,hi[0],hi[1]],lo)
    lower=borrowed_increment(lo,hi)
    return carry+upper_top+upper_rest+x(work)+carry+lower


def expand(gates):
    out=[]
    for name,qs,angle in gates:
        out+=exact_ccx(*qs) if name=='ccx' else [(name,list(qs),angle)]
    return out


def inventory(gates):
    c=Counter(name for name,_,_ in gates)
    return {'RZ':c['rz'],'native_T':c['t']+c['tdg'],
            'CX':c['cx'],'Clifford':sum(v for k,v in c.items() if k not in ['rz','t','tdg','ccx']),
            'unexpanded_CCX':c['ccx'],'by_gate':dict(c)}


def depth(gates,nqubits=9):
    last=[0]*nqubits
    for _,qs,_ in gates:
        d=1+max(last[q] for q in qs)
        for q in qs:last[q]=d
    return max(last)
