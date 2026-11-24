"""Check one-clean arithmetic, relative phases, flags and transport equivalence.

Run: python -m Tests.verify_one_clean
Exhaustive clean-input sectors cover every MCX and 6/8-bit shift used by the
selected grids. A coupled small-grid step checks the composed compiler too.
"""
from pathlib import Path
from collections import Counter
import json
import numpy as np
from Source.KvN import one_clean_synthesis as O, wake_gate_core as C, count_polar as P
from Source.ROM.model import load

ROOT=Path(__file__).resolve().parents[1]

def sector(data_bits, total_bits):
    d=1<<data_bits
    a=np.zeros((1<<total_bits,d),complex)
    a[np.arange(d),np.arange(d)]=1
    return a

def maxerr(a,b):return float(np.max(np.abs(a-b)))

def legacy_mcx(controls,target,work):
    m=len(controls)
    if m==0:return [('x',[target],None)]
    if m==1:return [('cx',[controls[0],target],None)]
    if m==2:return C.ccx(*controls,target)
    triples=[(controls[0],controls[1],work[0])]
    for i in range(2,m-1):triples.append((work[i-2],controls[i],work[i-1]))
    middle=(work[m-3],controls[-1],target)
    return sum((C.ccx(*v) for v in triples),[])+C.ccx(*middle)+sum((C.ccx(*v) for v in reversed(triples)),[])

def main():
    checks={'MCX':[],'shifts':[],'boundary_guards':[],'coupled_steps':[]}
    for m in range(8):
        # All basis columns, including both target values. Linearity therefore
        # also checks coherent superpositions and arbitrary relative phases.
        t=m;w=m+1;n=m+2;data=sector(m+1,n)
        gates=C.mcx(list(range(m)),t,[w]);got=C.apply_gates(data,gates,n)
        idx=np.arange(1<<(m+1));wanted=idx^((((idx&((1<<m)-1))==((1<<m)-1)).astype(int))<<t)
        expected=np.zeros_like(data);expected[wanted,idx]=1
        e=maxerr(got,expected);leak=float(np.linalg.norm(got[1<<(m+1):]))
        assert e<3e-13 and leak<3e-12,(m,e,leak)
        if m>=3:assert Counter(g[0] for g in gates)['t']+Counter(g[0] for g in gates)['tdg']==7*(2*m-3)
        checks['MCX'].append({'controls':m,'maximum_complex_error':e,'clean_work_leakage':leak})
    for n in (6,8):
        data=sector(n,n+1);inc=C.increment(n,[n]);idx=np.arange(1<<n)
        expected=np.zeros_like(data);expected[(idx+1)%(1<<n),idx]=1
        actual=C.apply_gates(data,inc,n+1)
        forward=maxerr(actual,expected)
        back=C.apply_gates(actual,C.inverse(inc),n+1)
        inverse=maxerr(back,data)
        assert max(forward,inverse)<8e-13,(n,forward,inverse)
        checks['shifts'].append({'register_bits':n,'columns':len(idx),'increment_error':forward,'inverse_error':inverse,'work_leakage':float(np.linalg.norm(actual[1<<n:]))})
        for span in (1,2):
            reg=P.Register(1<<n,0,0.,1.)
            shift,guard,_=P.matching_guard(reg,span,[n],n+1)
            g=shift+guard;initial=sector(n,n+2);got=C.apply_gates(initial,g,n+2)
            moved=(idx-span)%(1<<n)
            bit=span.bit_length()-1;mask=((1<<n)-1)^((1<<(bit+1))-1)
            flag=((moved&mask)==mask).astype(int)
            out=moved|(flag<<(n+1));expected=np.zeros_like(initial);expected[out,idx]=1
            err=maxerr(got,expected);restored=C.apply_gates(got,C.inverse(g),n+2)
            restore_error=maxerr(restored,initial)
            assert max(err,restore_error)<1e-12,(n,span,err,restore_error)
            checks['boundary_guards'].append({'register_bits':n,'span':span,'maximum_complex_error':err,'uncomputation_error':restore_error})
    rng=np.random.default_rng(20261117);model=load('candidate')
    original=C.mcx
    for stencil in (2,4):
        regs,pos,work,flags=P.settings((32,4,4),(0.,1.65),(.3,1.7))
        layers=P.schedule(stencil,'rkz')
        schedule=[(a,.5) for a in layers[:-1]]+[(layers[-1],1.)]+[(a,.5) for a in layers[-2::-1]]
        new=[g for a,scale in schedule for g in P.compile_layer(a,regs,model.coefficients,.017*scale,stencil,'face',work,flags)]
        legacy_work=list(range(pos,pos+2));legacy_flags=[pos+2,pos+3]
        try:
            C.mcx=legacy_mcx
            old=[g for a,scale in schedule for g in P.compile_layer(a,regs,model.coefficients,.017*scale,stencil,'face',legacy_work,legacy_flags)]
        finally:C.mcx=original
        z=rng.normal(size=1<<pos)+1j*rng.normal(size=1<<pos);z/=np.linalg.norm(z)
        a=np.zeros(1<<(pos+3),complex);a[:1<<pos]=z
        b=np.zeros(1<<(pos+4),complex);b[:1<<pos]=z
        anew=C.apply_gates(a,new,pos+3);bold=C.apply_gates(b,old,pos+4)
        err=maxerr(anew[:1<<pos],bold[:1<<pos]);leak=max(float(np.linalg.norm(anew[1<<pos:])),float(np.linalg.norm(bold[1<<pos:])))
        assert err<1e-12 and leak<2e-12,(stencil,err,leak)
        checks['coupled_steps'].append({'stencil_order':stencil,'position_bits':pos,'maximum_complex_error':err,'all_work_and_flag_leakage':leak})
    checks['status']='PASS'
    out=ROOT/'Build/One_Clean_Three_Coordinate/primitive_and_step_validation.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
